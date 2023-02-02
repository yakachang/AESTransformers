import json
import os
import re
from collections import Counter
from typing import Dict, List

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Text2TextGenerationPipeline,
)

BATCH_SIZE = 10

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

INPUT_TEMPLATE = """question: which description describes the word " {0} " best in the following context? \
descriptions:[  " {1} ",  or " {2} " ]
context: {3}"""

# PATH_TO_CAM_DATA = "../data"
PATH_TO_CAM_DATA = "/home/nlplab/kedy/NLP/AES/data"
FILE_CAM_DICT = "cambridge.word.888.json"

# Load Cambridge Dict
with open(os.path.join(PATH_TO_CAM_DATA, FILE_CAM_DICT), "r") as f:
    cambridge_dict = json.load(f)

"""
Model Info
* T5-small
* SemCor in NLTK Corpus
"""
PATH_TO_MODEL = "../../AESbackend/prep/wsd/models/baseline-1-t5-small-opt/"
model = AutoModelForSeq2SeqLM.from_pretrained(PATH_TO_MODEL)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_MODEL)

pipe = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)


def generate_word_level_candidates(token, sent, vocab_levels: List, wsd_list: List):
    """
    Generate word level candidates
    :param token: spacy token object
    :param sent: spacy sent object
    :param vocab_levels: List
    :param wsd_list: List
    :return: None
    """

    word_info = cambridge_dict.get(token.lemma_, None)
    if word_info:
        definitions = []
        pos_list = list(word_info.keys())
        if len(pos_list):
            for pos in pos_list:
                for item1 in word_info[pos]:
                    for item2 in item1["big_sense"]:
                        for sense_block in item2["sense"]:
                            if sense_block["level"] in CEFR_LEVELS:
                                definitions.append(
                                    {
                                        "def": sense_block["en_def"].lower(),
                                        "level": sense_block["level"],
                                    }
                                )

        if len(definitions) == 1:
            vocab_levels.append(definitions[0]["level"])

        elif len(definitions) > 1:
            for i, item in enumerate(definitions):
                definitions[i] = {
                    "def": f"({i + 1}) {item['def']}",
                    "level": item["level"],
                }

            input = INPUT_TEMPLATE.format(
                token.text,
                ' " , " '.join([item["def"] for item in definitions][:-1]),
                [item["def"] for item in definitions][-1],
                sent.text.replace(token.text, f'" {token.text} "'),
            )

            wsd_list.append({"input": input, "definitions": definitions})


def wsd(wsd_list: List, vocab_levels: List) -> Dict:
    """
    Do word sense disambiguation to extract level of vocab in specific sentence
    :param wsd_list: List
    :param vocab_levels: List
    :return: str
    """

    inputs = [item["input"] for item in wsd_list if "input" in item]
    indexes = [result["generated_text"] for result in pipe(inputs)]

    definitions_list = [
        item["definitions"] for item in wsd_list if "definitions" in item
    ]

    for index, definitions in zip(indexes, definitions_list):
        try:
            idx = int(re.search(r"(\d+)", index).group(1)) - 1
            vocab_levels.append(definitions[idx]["level"])
        except Exception as e:
            print({"error_msg": e, "index": index, "idx": idx, "defs": definitions})

    counter = Counter(vocab_levels)

    level_dist = {}
    summation = sum(dict(counter).values())
    for key, value in counter.items():
        level_dist[key] = value / summation

    return level_dist


def get_vocab_level_dist(doc, STOP_WORDS):
    vocab_levels, wsd_list = [], []

    for sent in doc.sents:
        for token in sent:
            if token.lemma_ in STOP_WORDS:
                continue

            generate_word_level_candidates(token, sent, vocab_levels, wsd_list)

    return wsd(wsd_list, vocab_levels)
