from spacy.tokens import Token


def is_verb(token: Token) -> bool:
    return token.tag_.startswith(("V", "M"))


def is_be_verb(token: Token) -> bool:
    return token.lemma_ == "be" and token.dep_ == "aux"


def is_passive_be(token: Token) -> bool:
    """Checks if the token is a passive BE-verb.

    Example:
    1. The problem will *be* resolved.
    2. The glass *is* broken.
    """
    return token.lemma_ == "be" and token.dep_ == "auxpass"


def is_ing(token: Token) -> bool:
    """Checks if the token is in gerund or present participle form."""
    return token.tag_ == "VBG"


def is_pp(token: Token, strict: bool = False) -> bool:
    """Checks if the token is in past participle form.

    Args:
        token: The target token to analyze.
        strick: allow past tense as well to prevent parsing error.
    """
    if token.tag_ == "VBN":
        return True
    if not strict and token.tag_ == "VBD":
        return True
    return False


def is_inf(token: Token) -> bool:
    """Checks if the token is an infinitive verb.
    Example: `I don't dare "say" that`, `go "see" what happened`, etc.
    """
    return token.tag_ == "VB"


def is_to_inf(token: Token) -> bool:
    """Checks if the token is an infinitive verb of `to-inf`.
    Example: `to *do*`, `what to *say*`, etc.
    """
    if not token.tag_ == "VB":
        return False

    return any(
        (child.text == "to" and child.tag_ == "TO" and child.dep_ == "aux")
        for child in token.lefts
    )


def is_inf_to(token: Token) -> bool:
    """Checks if the token is a 'to' in an infinitive.
    Example: `*to* do`, `what *to* say`, etc.
    """
    return token.lemma_ == "to" and token.tag_ == "TO" and is_inf(token.head)


def is_prep(token: Token, check_dep: bool = True) -> bool:
    return (
        token.tag_ == "IN" and token.dep_ == "prep" if check_dep else token.tag_ == "IN"
    )


def is_adj(token: Token, include_comparative: bool = True) -> bool:
    if include_comparative:
        return token.tag_.startswith("J")
    return token.tag_ == "JJ"


def is_adv(token: Token, include_particle: bool = True) -> bool:
    if include_particle:
        return token.tag_.startswith("R")
    return token.tag_.startswith("RB")


def is_noun(token: Token) -> bool:
    return token.tag_.startswith(("NN", "PR", "PO"))


def is_pln(token: Token, strict: bool = False) -> bool:
    """Checks if the given token is a plural noun."""
    if token.tag_ in ["NNPS", "NNS"]:
        return True
    if token.lemma_ in ["we", "us", "they", "them"]:
        return True
    if strict:
        return False

    if token.tag_ == "NN":
        # This includes the mass noun, like fish, team, family, etc.
        return True
    if token.lemma_ == "you":
        return True
    return False
