from spacy.tokens import Token

from .token_checker import is_adv, is_inf_to, is_passive_be, is_pp
from .utility import simply_tag, sort_tokens


def is_related_token(root: Token, child: Token, grammar_tag: str):
    root_tag = simply_tag(root.tag_)
    child_dep = child.dep_

    if root_tag == "n":
        if child_dep in [
            "det",
            "compound",
            "amod",
            "nmod",
            "nounmod",
            "nummod",
            "quantmod",
            "advmod",
            "poss",
        ]:
            return True

    if root_tag == "v":
        if is_pp(root) and is_passive_be(child):
            return True
        if is_inf_to(child):
            return False
        if is_adv(child) and child_dep == "advmod":
            return True
        if child_dep in ["neg", "aux"]:
            return True

    if root_tag == "adj":
        if is_adv(child) and child_dep == "advmod":
            return True

    return False


def extract_chunk(root: Token, grammar_tag: str):
    """Extracts a meaningful phrase chunk from a given token."""

    stack = [*root.children]
    extracted = [root]
    while len(stack) > 0:
        child = stack.pop(0)
        if is_related_token(root, child, grammar_tag):
            extracted.append(child)
            stack.extend([*child.children])
    return sort_tokens(extracted)
