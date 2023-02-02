import json
import os
import string
from collections import Counter
from typing import Dict

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

PATH_TO_CAM_DATA = "../data"
FILE_CAM_DICT = "cambridge.word.888.json"

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR2NUM = {level: idx for idx, level in enumerate(CEFR_LEVELS)}
NUM2CEFR = {idx: level for idx, level in enumerate(CEFR_LEVELS)}

STOP_WORDS = set(stopwords.words("english"))

with open(os.path.join(PATH_TO_CAM_DATA, FILE_CAM_DICT), "r") as f:
    cambridge_dict = json.load(f)

model = AutoModelForSeq2SeqLM.from_pretrained("jpwahle/t5-word-sense-disambiguation")
tokenizer = T5Tokenizer.from_pretrained("jpwahle/t5-word-sense-disambiguation")

INPUT_TEMPLATE = """question: which description describes the word " {0} " best in the following context? \
descriptions:[  " {1} ",  or " {2} " ]
context: {3}"""


def calculate_vocabs_word_count(text: str) -> int:

    words = []

    sents = sent_tokenize(text)
    for sent in sents:
        tokens = word_tokenize(sent)
        words.extend(tokens)

    words = [word for word in words if word not in string.punctuation]

    return len(set(words)), len(words)


def analysis_level_distribution(text: str) -> str:

    vocab_levels = []
    wsd_list = []
    sents = sent_tokenize(text)
    for sent in sents:
        tokens = word_tokenize(sent)
        for token in tokens:
            if token in STOP_WORDS:
                continue

            word_info = cambridge_dict.get(token, None)
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
                        token,
                        ' " , " '.join(list(definitions.keys())[:-1]),
                        list(definitions.keys())[-1],
                        sent.replace(token, f'" {token} "'),
                    )

                    wsd_list.append({"input": input, "definitions": definitions})

                    # input_ids = tokenizer(input, return_tensors="pt").input_ids
                    # answer = model.generate(input_ids)
                    # best_def = tokenizer.decode(answer[0], skip_special_tokens=True)
                    # if best_def in definitions:
                    #     vocab_levels.append(definitions[best_def])
                    # else:
                    #     # TODO: Deal with best_def not found in definitions
                    #     for definition in list(definitions.keys()):
                    #         if best_def in definition:
                    #             vocab_levels.append(definitions[definition])

    inputs = [item["input"] for item in wsd_list if "input" in item][:3]
    definitions_list = [
        item["definitions"] for item in wsd_list if "definitions" in item
    ][:3]
    input_ids = tokenizer(
        inputs,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids
    answers = model.generate(input_ids)
    best_defs = tokenizer.batch_decode(answers, skip_special_tokens=True)

    for best_def, definitions in zip(best_defs, definitions_list):
        if best_def in definitions:
            vocab_levels.append(definitions[best_def])
        else:
            # TODO: Deal with best_def not found in definitions
            for definition in list(definitions.keys()):
                if best_def in definition:
                    vocab_levels.append(definitions[definition])

    counter = Counter(vocab_levels)

    distribution = ""
    for level in CEFR_LEVELS:
        distribution += "{0}: {1:.0%} ".format(
            level, float(counter[level]) / sum(counter.values())
        )

    return distribution


def report(text: str) -> Dict:

    vocab_num, word_count = calculate_vocabs_word_count(text)

    return {
        "vocab_num": vocab_num,
        "word_count": word_count,
        "level_distribution": analysis_level_distribution(text),
    }
