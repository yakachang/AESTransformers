import re
from collections import Counter
from typing import Dict

import errant
import spacy
from errant.edit import Edit
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REGEX = re.compile(r"\:([A-Z]+)")

CODE2CATEGORY = {
    "ADJ": "Collocation",
    "ADV": "Collocation",
    "NOUN": "Collocation",
    "VERB": "Collocation",
    "FORM": "Form",
    "TENSE": "Form",
    "SVA": "Form",
    "NUM": "Form",
    "CONJ": "Conjunction",
    "PART": "Preposition",
    "PREP": "Preposition",
    "PUNCT": "Punctuation",
    "SPELL": "Spelling",
    "WO": "Word Order",
    "CONTR": "Other",
    "DET": "Other",
    "PRON": "Other",
    "MORPH": "Other",
    "NA": "Other",
    "ORTH": "Other",
    "OTHER": "Other",
}

nlp = spacy.load("en_core_web_sm")
annotator = errant.load("en", nlp)

MODEL_PATH = "/home/nlplab/jjc/gec/t5-small-clean-new"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)


def correct(sents, prefix="gec: "):
    # prepare data as the input format of gec model
    sents = [prefix + sent for sent in sents]
    input_ids = tokenizer(
        sents, return_tensors="pt", padding=True, truncation=True
    ).input_ids
    # set max_length as size plus a number in case of insertion edits
    max_length = min(input_ids.size(1) + 10, tokenizer.model_max_length)
    outputs = model.generate(input_ids, max_length=max_length)
    res = [
        tokenizer.decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for output in outputs
    ]
    # TODO: add back truncated text
    return res


def generate_edit_info(edit, offset=0):
    return {
        "start": edit.o_toks.start_char + offset,
        "end": edit.o_toks.end_char + offset,
        "correction": edit.c_str,
        "type": edit.type,
    }


def normalize_insertion(edit):
    # the start index is wrong when inserting in the end
    if edit.o_start >= len(edit.o_toks.doc):
        edit_span = (
            max(edit.o_start - 1, 0),
            edit.o_end,
            max(edit.c_start - 1, 0),
            edit.c_end,
        )
    else:
        edit_span = (edit.o_start, edit.o_end + 1, edit.c_start, edit.c_end + 1)
    return Edit(edit.o_toks.sent, edit.c_toks.sent, edit_span)


def merge_overlap_edits(edits):
    edit_indexs = set()
    edit_pools = []
    for edit in edits:
        if edit_pools and edit.o_start in edit_indexs:
            edit_pools[-1].append(edit)
        else:
            edit_pools.append([edit])
        edit_indexs.update(range(edit.o_start, edit.o_end))

    for edits in edit_pools:
        edit_span = (
            edits[0].o_start,
            edits[-1].o_end,
            edits[0].c_start,
            edits[-1].c_end,
        )
        err_type = ",".join(edit.type for edit in edits)
        yield Edit(edits[0].o_toks.doc, edits[0].c_toks.doc, edit_span, err_type)


def extract_edits(orig, cor, offset=0):
    edits = annotator.annotate(orig, cor)
    # convert insertion into replacement
    edits = [edit if edit.o_toks else normalize_insertion(edit) for edit in edits]
    # merge overlap edits
    return [generate_edit_info(edit, offset) for edit in merge_overlap_edits(edits)]


def translate_type(type: str) -> str:

    if len(REGEX.findall(type)):
        return CODE2CATEGORY.get(REGEX.findall(type)[-1], "Other")

    return "Other"


def tokenize_text(text: str):

    doc = nlp(text)

    return " ".join([token.text for sent in doc.sents for token in sent])


def gec(text: str) -> Dict:

    corr = correct([text])[0]

    text_parsed = annotator.parse(tokenize_text(text))
    corr_parsed = annotator.parse(tokenize_text(corr))
    result = extract_edits(text_parsed, corr_parsed)

    grammar_error_list = [translate_type(item["type"]) for item in result]

    return dict(Counter(grammar_error_list))
