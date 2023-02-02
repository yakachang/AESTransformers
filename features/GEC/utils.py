import errant
from errant.edit import Edit
from nltk.tokenize import load

# set `nlp` to True to not load spacy models since parsing is not used
annotator = errant.load("en", nlp=True)

ERROR_TYPES = {
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
    "SPACE": "SPACE",
    "OTHER": "Other",
}


def sent_tokenize(text, language="english"):
    tokenizer = load(f"tokenizers/punkt/{language}.pickle")
    return tuple(tokenizer.span_tokenize(text))


def generate_edit_info(edit, offset=0):
    return {
        "start": edit.o_toks.start_char + offset,
        "end": edit.o_toks.end_char + offset,
        "correction": edit.c_str,
        "code": edit.type,
        "type": ERROR_TYPES.get(edit.type.split(":")[-1], "Other"),
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
