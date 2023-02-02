import json
import os
from collections import Counter
from typing import Dict, List

from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, Text2TextGenerationPipeline

BATCH_SIZE = 32

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

INPUT_TEMPLATE = """question: which description describes the word " {0} " best in the following context? \
descriptions:[  " {1} ",  or " {2} " ]
context: {3}"""

PATH_TO_CAM_DATA = "../data"
FILE_CAM_DICT = "cambridge.word.888.json"

# Load Cambridge Dict
with open(os.path.join(PATH_TO_CAM_DATA, FILE_CAM_DICT), "r") as f:
    cambridge_dict = json.load(f)

"""
Model Info
* https://huggingface.co/jpwahle/t5-word-sense-disambiguation
* T5-large
* SemCor 3.0 dataset
"""
model = AutoModelForSeq2SeqLM.from_pretrained("jpwahle/t5-word-sense-disambiguation")
tokenizer = T5Tokenizer.from_pretrained("jpwahle/t5-word-sense-disambiguation")

pipe = Text2TextGenerationPipeline(
    model=model, tokenizer=tokenizer, batch_size=BATCH_SIZE, device=1
)


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
        definitions = {}
        pos_list = list(word_info.keys())
        if len(pos_list):
            for pos in pos_list:
                for item1 in word_info[pos]:
                    for item2 in item1["big_sense"]:
                        for sense_block in item2["sense"]:
                            if sense_block["level"] in CEFR_LEVELS:
                                definitions[
                                    sense_block["en_def"].lower()
                                ] = sense_block["level"]

        if len(definitions) == 1:
            vocab_levels.append(list(definitions.values())[0])

        elif len(definitions) > 1:
            input = INPUT_TEMPLATE.format(
                token.text,
                ' " , " '.join(list(definitions.keys())[:-1]),
                list(definitions.keys())[-1],
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
    definitions_list = [
        item["definitions"] for item in wsd_list if "definitions" in item
    ]

    best_defs = [result["generated_text"] for result in pipe(inputs)]

    for best_def, definitions in zip(best_defs, definitions_list):
        if best_def in definitions:
            vocab_levels.append(definitions[best_def])
        else:
            # TODO: Deal with best_def not found in definitions
            for definition in list(definitions.keys()):
                if best_def in definition:
                    vocab_levels.append(definitions[definition])

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
