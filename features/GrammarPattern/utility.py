import logging
from time import process_time_ns


def sort_tokens(tokens):
    """Sorts spacy tokens by indice"""
    return sorted(tokens, key=lambda token: token.i)


def unpack(pack_list):
    # TODO: Double-check!
    # try:
    #     merged = [unpack(pack) for pack in pack_list]
    # except:
    #     return [pack_list]
    merged = [unpack(pack) for pack in pack_list]
    return sum(merged, [])


def print_dict(dict_object):
    """Prints dict with better format"""
    print("{", end="")
    first = True
    for k, v in dict_object.items():
        if first:
            first = False
        else:
            print(",\n ", end="")
        print(f" {k!r}: {v!r}", end="")
    print(" }")


def simply_tag(tag):
    """Gets simplified token tag"""
    tag = tag.lower()
    if tag in [
        "noun",
        "propn",
        "pron",  # spacy pos
        "nn",
        "nnp",
        "nnps",
        "nns",
        "prp",
        "prp$",  # spacy tag
        "pl-n",  # collins tag
    ]:
        return "n"
    if tag in [
        "verb",
        "aux",  # spacy pos
        "vb",
        "vbd",
        "vbg",
        "vbn",
        "vbp",
        "vbz",
        "md",  # spacy tag
        "v-link",  # collins tag
    ]:
        return "v"
    if tag in [
        "adj",  # spacy pos
        "jj",
        "jjr",
        "jjs" "adj",  # spacy tag  # collins tag
    ]:
        return "adj"
    return tag


def elapse(func):
    def wrapper(*args, **kwargs):
        _time = process_time_ns()
        ret = func(*args, **kwargs)
        logging.debug(f"{func.__name__}() -> {process_time_ns() - _time}")
        return ret

    return wrapper
