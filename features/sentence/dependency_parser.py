import numpy as np


def tree_height(root) -> int:
    """
    Find the maximum depth (height) of the dependency parsed sentence by starting with its root.
    :param root: spacy.tokens.token.Token
    :return: int, maximum height of sentence's dependency parse tree
    """
    if not list(root.children):
        return 1
    else:
        return 1 + max(tree_height(x) for x in root.children)


def get_average_heights(doc) -> float:
    """
    Computes average height of parse trees for each sentence in paragraph.
    :param paragraph: spacy doc object
    :return: float
    """
    roots = [sent.root for sent in doc.sents]

    return np.mean([tree_height(root) for root in roots])
