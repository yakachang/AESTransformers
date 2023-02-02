import logging
from typing import List, Optional, Set

from spacy.tokens import Token

from . import utility
from .PatternHolder import PatternHolder
from .Result import MatchedPattern
from .token_checker import (
    is_adj,
    is_adv,
    is_inf,
    is_ing,
    is_noun,
    is_passive_be,
    is_pln,
    is_pp,
    is_prep,
    is_verb,
)


class PatternValidator(object):
    """Holds all the pattern grammar, and extracts the informations for matching patterns.

    Args:
        pattern_dir: the path to the pattern data.
        logger: the logging object with log API such as `error()`, `debug()`, etc.
    """

    def __init__(
        self,
        pattern: Optional[PatternHolder] = None,
        pattern_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger()
        self.pattern = pattern or PatternHolder(data_dir=pattern_dir)

    # ------ COLLINS GRAMMAR PATTERN ------ #
    # VERB #
    def V_n_prep_n(
        self, token: Token, prep: str, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        if not is_verb(token):
            return

        pattern = f"V n {prep} n"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []
        """
        1. find object n1
           - passive
           - active
        2. find prep
        3. find n2
           - object
           - clause
           - complement
        """

        # find passive n (passive subject)
        S, is_passive_voice = None, False
        if is_pp(token):
            self.logger.debug("checking for passive voice")
            for child in token.lefts:
                if child.is_punct:
                    continue

                if is_passive_be(child):
                    is_passive_voice = True
                    break
                if child.dep_ == "nsubjpass":
                    self.logger.debug(f"found passive subject {child.text!r}")
                    S = child
                    continue

            if not is_passive_voice:
                self.logger.debug(
                    "No BE-verb is found. Stop checking for passive voice."
                )
                S = None
            elif not S and token.dep_ == "xcomp":
                for child in token.head.lefts:
                    if child.is_punct:
                        continue

                    if child.dep_ == "nsubj":
                        self.logger.debug(f"found passive subject {child.text!r}")
                        S = child
                        continue

            if not S:
                self.logger.debug(
                    "Passive subject not found. Stop checking for passive voice."
                )
                is_passive_voice = False

        n1, p, n2 = S, None, None
        for child in token.rights:
            if child.is_punct:
                continue

            # find n1
            if not n1:
                if child.dep_ == "dobj":
                    self.logger.debug(f"found n1 {child.text!r}")
                    n1 = child
                continue

            # find prep from token
            if child.lemma_ == prep and child.dep_ in ["prep", "agent", "dative"]:
                self.logger.debug(f"found preposition {child.text!r}")
                p = child
                break
        if not n1:
            self.logger.debug("n1 not found. abort.")
            return

        # in case prep is the child of n1
        if not p:
            self.logger.debug(f"try to find preposition from n1 {n1.text!r}")
            for child in n1.rights:
                if child.is_punct:
                    continue

                if child.lemma_ == prep and child.dep_ in ["prep", "agent", "dative"]:
                    self.logger.debug(f"found preposition {child.text!r}")
                    p = child
                    break
        if not p:
            self.logger.debug("preposition not found. abort.")
            return

        # find n2
        for child in p.rights:
            if child.is_punct:
                continue

            if child.dep_ == "pobj":
                self.logger.debug(f"found noun as n2 {child.text!r}")
                n2 = child
            if (
                prep
                in [
                    "about",
                    "against",
                    "as",
                    "by",
                    "for",
                    "from",
                    "in",
                    "of",
                    "on",
                    "to",
                    "toward",
                    "towards",
                    "with",
                ]
                and child.dep_ == "pcomp"
            ):
                self.logger.debug(f"found complement as n2 {child.text!r}")
                n2 = child

            if n2:
                results.append(
                    MatchedPattern(
                        pattern=["V", "n", prep, "n"],
                        tag=["Headword", "Content", "Grammar", "Content"],
                        token=[(token,), (n1,), (p,), (n2,)],
                        passive=is_passive_voice,
                    )
                )
                break

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_n_prep(
        self, token: Token, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        if not is_verb(token):
            return

        pattern = "V n prep"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        S, is_passive_voice = None, False
        if is_pp(token):
            self.logger.debug("checking for passive voice")
            for child in token.lefts:
                if child.is_punct:
                    continue

                if is_passive_be(child):
                    is_passive_voice = True
                    break
                if child.dep_ == "nsubjpass":
                    self.logger.debug(f"found passive subject {child.text!r}")
                    S = child
                    continue

            if not is_passive_voice:
                self.logger.debug(
                    "No BE-verb is found. Stop checking for passive voice."
                )
                S = None
            if not S:
                self.logger.debug(
                    "Passive subject no found. Stop checking for passive voice."
                )
                is_passive_voice = False

        n, prep = S, None
        for child in token.rights:
            # find n
            if not n:
                if child.dep_ == "dobj":
                    self.logger.debug(f"found n {child.text!r}")
                    n = child
                continue

            # find prep
            if child.tag_ == "IN" and child.dep_ in [
                "prep",
                "prt",
                "agent",  # passive
            ]:
                self.logger.debug(f"match preposition {child.text!r}")
                prep = child
            if not prep:
                continue

            results.append(
                MatchedPattern(
                    pattern=["V", "n", "prep"],
                    tag=["Headword", "Content", "Grammar"],
                    token=[(token,), (n,), (prep,)],
                    passive=is_passive_voice,
                )
            )
            break

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_n_adv(
        self, token: Token, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        if not is_verb(token):
            return

        pattern = "V n adv"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        S, is_passive_voice = None, False
        if is_pp(token):
            self.logger.debug("checking for passive voice")
            for child in token.lefts:
                if child.is_punct:
                    continue

                if is_passive_be(child):
                    is_passive_voice = True
                    break
                if child.dep_ == "nsubjpass":
                    self.logger.debug(f"found passive subject {child.text!r}")
                    S = child
                    continue

            if not is_passive_voice:
                self.logger.debug(
                    "No BE-verb is found. Stop checking for passive voice."
                )
                S = None
            if not S:
                self.logger.debug(
                    "Passive voice not found. Stop checking for passive voice."
                )
                is_passive_voice = False

        n, adv = S, None
        for child in token.rights:
            if child.is_punct:
                continue

            # Note that the order of n and adv is interchangable

            if not n and child.dep_ == "dobj":
                self.logger.debug(f"found n {child.text!r}")
                n = child
            if is_adv(child) and child.dep_ in ["advmod", "prt"]:
                self.logger.debug(f"match adv {child.text!r}")
                adv = child

            if n and adv:
                results.append(
                    MatchedPattern(
                        pattern=["V", "n", "adv"],
                        tag=["Headword", "Content", "Grammar"],
                        token=[(token,), (n,), (adv,)],
                        passive=is_passive_voice,
                    )
                )
                break

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_n_that(
        self, token: Token, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        if not is_verb(token):
            return

        pattern = "V n that"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []
        """
        1. find n
           - find passive n (passive subject)
           - find active n (object)
        2. find clause
        """

        # find passive n (passive subject)
        S, is_passive_voice = None, False
        if is_pp(token):
            self.logger.debug("checking for passive voice")
            for child in token.lefts:
                if child.is_punct:
                    continue

                if is_passive_be(child):
                    self.logger.debug(f"found passive BE {child.text!r}")
                    is_passive_voice = True
                    break
                if child.dep_ == "nsubjpass":
                    S = child
                    self.logger.debug(f"found passive subject {child.text!r}")
                    continue

            if not is_passive_voice:
                self.logger.debug(
                    "No BE-verb is found. Stop checking as passive voice."
                )
                S = None
            elif not S:
                self.logger.debug(
                    "Passive subject not found. Stop checking as passive voice."
                )
                is_passive_voice = False

        self.logger.debug("checking for that-clause or n + that-clause")
        n = S
        for child in token.rights:
            if child.is_punct:
                continue

            # find active n (object)
            if child.dep_ == "dobj":
                self.logger.debug(f"found object {child.text!r}")
                n = child
            if not n:
                continue

            # find clause
            matched = False
            if child.dep_ == "ccomp" and (
                that_clauses := self.clause(child, skip_wh=True, skip_comp=True)
            ):
                self.logger.debug(
                    f"found candidate complements from root {child.text!r}"
                )
                for clause in that_clauses:
                    that = None
                    if clause.pattern[0] == "that":
                        that = clause.token[0]
                    else:  # `that` can be omitted
                        that = tuple()

                    results.append(
                        MatchedPattern(
                            pattern=["V", "n", "that"],
                            tag=["Headword", "Content", "Grammar"],
                            token=[(token,), (n,), that],
                            passive=is_passive_voice,
                        )
                    )
                    matched = True

            if matched:
                break

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_n_ing(
        self, token: Token, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        if not is_verb(token):
            return

        pattern = "V n -ing"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        S, is_passive_voice = None, False
        if is_pp(token):
            self.logger.debug("checking for passive voice")
            for child in token.lefts:
                if child.is_punct:
                    continue

                if is_passive_be(child):
                    self.logger.debug(f"found passive BE {child.text!r}")
                    is_passive_voice = True
                    break
                if child.dep_ == "nsubjpass":
                    self.logger.debug(f"found passive subject {child.text!r}")
                    S = child
                    continue

            if not is_passive_voice:
                self.logger.debug(
                    "No BE-verb is found. Stop checking as passive voice."
                )
                S = None
            elif not S:
                self.logger.debug(
                    "Passive subject not found. Stop checking as passive voice."
                )
                is_passive_voice = False

        self.logger.debug("checking for Ving or n V-ing")
        n = S
        for child in token.rights:
            if child.is_punct:
                continue

            # find V-ing
            if n:
                if is_ing(child) and child.dep_ in ["xcomp", "advcl"]:
                    self.logger.debug(f"found V-ing {child.text!r}")
                    results.append(
                        MatchedPattern(
                            pattern=["V", "n", "-ing"],
                            tag=["Headword", "Content", "Content"],
                            token=[(token,), (n,), (child,)],
                            passive=is_passive_voice,
                        )
                    )
                    break
                else:
                    self.logger.debug(
                        f"Ving not immediately following the object. N {n.text!r} ignored."
                    )
                    if is_passive_voice:
                        break
                    else:
                        n = None

            # find N
            if child.dep_ == "dobj":
                self.logger.debug(f"found object {child.text!r}")
                n = child
                continue

            # find 'n V-ing' from root 'V-ing'
            if is_ing(child) and child.dep_ == "ccomp":
                self.logger.debug(f"found V-ing {child.text!r}")
                for subchild in child.lefts:
                    if subchild.is_punct:
                        continue

                    if subchild.dep_ == "nsubj":
                        self.logger.debug(f"found n {subchild.text!r}")
                        results.append(
                            MatchedPattern(
                                pattern=["V", "n", "-ing"],
                                tag=["Headword", "Content", "Content"],
                                token=[(token,), (subchild,), (child,)],
                                passive=False,
                            )
                        )
                        break

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_n_n(
        self, token: Token, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        if not is_verb(token):
            return

        pattern = "V n n"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        S, is_passive_voice = None, False
        if is_pp(token):
            self.logger.debug("checking for passive voice")
            for child in token.lefts:
                if child.is_punct:
                    continue

                if is_passive_be(child):
                    self.logger.debug(f"found passive BE {child.text!r}")
                    is_passive_voice = True
                    break
                if child.dep_ == "nsubjpass":
                    self.logger.debug(f"found passive subject {child.text!r}")
                    S = child
                    continue

        if not is_passive_voice:
            self.logger.debug("No BE-verb is found. Stop checking as passive voice.")
            S = None
        elif not S:
            self.logger.debug(
                "Passive subject not found. Stop checking as passive voice."
            )
            is_passive_voice = False

        self.logger.debug("checking for nouns")
        n1, n2 = S, None
        for child in token.rights:
            if child.is_punct:
                continue

            if child.dep_ in ["dative", "oprd", "dobj", "npadvmod"]:
                if n1:
                    self.logger.debug(f"found n2 {child.text!r}")
                    n2 = child
                else:
                    self.logger.debug(f"found n1 {child.text!r}")
                    n1 = child
                    continue

            if n1 and n2:
                results.append(
                    MatchedPattern(
                        pattern=["V", "n", "n"],
                        tag=["Headword", "Content", "Content"],
                        token=[(token,), (n1,), (n2,)],
                        passive=is_passive_voice,
                    )
                )
                break

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_prep_n(
        self, token: Token, prep: str, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        """
        Available p = [about, across, after, against, around, round, as, at, for, from, in, into,
                       like, of, off, on, onto, over, through, to, towards, toward, under, with]
        No passive voice: [across, around, round, as, like, onto, toward, towards]
        """
        if not is_verb(token):
            return

        pattern = f"V {prep} n"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")

        # find the preposition
        prep_token = None
        for child in token.rights:
            if child.is_punct:
                continue

            if child.text == prep and child.dep_ in ["prep", "advmod", "prt"]:
                prep_token = child
                break

        if not prep_token:
            self.logger.debug(f"preposition {prep!r} not found")
            return

        results = []
        is_passive_voice = False
        if is_pp(token) and prep not in [
            "across",
            "around",
            "round",
            "as",
            "like",
            "onto",
            "toward",
            "towards",
        ]:
            self.logger.debug("checking for passive voice")
            for child in token.lefts:
                if child.is_punct:
                    continue
                if is_passive_be(child):
                    is_passive_voice = True
                    break

                if child.dep_ == "nsubjpass":
                    self.logger.debug(f"found passive noun {child.text!r}")
                    results.append(
                        MatchedPattern(
                            pattern=["V", prep, "n"],
                            tag=["Headword", "Grammar", "Content"],
                            token=[(token,), (prep_token,), (child,)],
                            passive=True,
                        )
                    )

                if child.dep_ == "csubjpass":
                    self.logger.debug(
                        f"found passive prep-phrase as noun {child.text!r}"
                    )
                    results.append(
                        MatchedPattern(
                            pattern=["V", prep, "n"],
                            tag=["Headword", "Grammar", "Content"],
                            token=[(token,), (prep_token,), (child,)],
                            passive=True,
                        )
                    )
            if not is_passive_voice:
                self.logger.debug("No BE-verb is found. All passive n are ignored.")
                results.clear()

        if not is_passive_voice:
            self.logger.debug("checking for active voice")
            has_found = False

            # check from children of prep first
            for child in prep_token.rights:
                if child.is_punct:
                    continue

                if child.dep_ in ["pobj", "pcomp"]:
                    self.logger.debug(f"found noun {child.text!r}")
                    results.append(
                        MatchedPattern(
                            pattern=["V", prep, "n"],
                            tag=["Headword", "Grammar", "Content"],
                            token=[(token,), (prep_token,), (child,)],
                            passive=False,
                        )
                    )
                    has_found = True
                    break

            # check for children of the token
            for child in token.rights:
                if has_found:
                    break
                if child.is_punct:
                    continue

                if is_noun(child) and child.dep_ == "dobj":
                    self.logger.debug(f"found dobj as noun {child.text!r}")
                    results.append(
                        MatchedPattern(
                            pattern=["V", prep, "n"],
                            tag=["Headword", "Grammar", "Content"],
                            token=[(token,), (prep_token,), (child,)],
                            passive=False,
                        )
                    )
                    has_found = True

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_adv(
        self, token: Token, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        if not is_verb(token):
            return

        pattern = "V adv"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        is_passive_voice = False
        if is_pp(token):
            self.logger.debug("checking for passive voice")
            for child in token.lefts:
                if child.is_punct:
                    continue

                if is_passive_be(child):
                    is_passive_voice = True
                    break

        if is_passive_voice:
            self.logger.debug(
                "This pattern should not be used in passive voice. abort."
            )
            return

        for child in token.rights:
            if is_adv(child) and child.dep_ in ["advmod", "prt"]:
                self.logger.debug(f"match adverb {token.text!r}")
                results.append(
                    MatchedPattern(
                        pattern=["V", "adv"],
                        tag=["Headword", "Grammar"],
                        token=[(token,), (child,)],
                        passive=False,
                    )
                )

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_prep(
        self, token: Token, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        if not is_verb(token):
            return

        pattern = "V prep"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        is_passive_voice = False
        if is_pp(token):
            self.logger.debug("checking for passive voice")
            for child in token.lefts:
                if child.is_punct:
                    continue

                if is_passive_be(child):
                    is_passive_voice = True
                    break

            if not is_passive_voice:
                self.logger.debug("No BE-verb is found. Not checking for passive prep.")

        for child in token.rights:
            if child.is_punct:
                continue

            if child.tag_ == "IN" and child.dep_ in ["prep", "prt", "agent"]:
                self.logger.debug(f"found preposition {token.text!r}")
                results.append(
                    MatchedPattern(
                        pattern=["V", "prep"],
                        tag=["Headword", "Grammar"],
                        token=[(token,), (child,)],
                        passive=is_passive_voice,
                    )
                )
                break

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_and_v(
        self, token: Token, strict_level: int = 1, check_pattern: bool = False
    ) -> Optional[List[MatchedPattern]]:
        """Checks for Collins pattern "V and v".
        Generally, any two or more verbs can be conjuncted with the word "and",
        but their tenses should be the same.

        Args:
            token: The root token to analyze.
            strick_level: How strick the check the tenses of conjuncted verbs.
              0: Don't check tenses at all.
              1: Allow to mix past tense and past particle,
                 3rd person and non-3rd person singular present.
              2: tags should exactly match.
            check_pattern: Whether to return verbs explictly having "V and v" pattern
                           in Collins Grammar Pattern.
        """

        def loosely_match(v1, v2):
            if v1.tag_ == v2.tag_:
                return True

            loose_groups = [
                ("VBD", "VBN"),
                ("VBP", "VBZ"),
            ]
            for group in loose_groups:
                if v1.tag_ in group and v2.tag_ in group:
                    return True
            return False

        if not is_verb(token):
            return

        pattern = "V and v"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        cconj = None  # and
        for child in token.rights:
            if child.is_punct:
                continue

            if child.lemma_ == "and" and child.dep_ == "cc":
                self.logger.debug(f"match coordinate conjunction {child.text!r}")
                cconj = child
                continue
            if not cconj:
                continue

            if child.dep_ == "conj":
                if strict_level == 0 and is_verb(child):
                    self.logger.debug(
                        f"match conj {child.text!r} with strick level {strict_level}"
                    )
                    results.append(
                        MatchedPattern(
                            pattern=["V", "and", "v"],
                            tag=["Headword", "Grammar", "Content"],
                            token=[(token,), (cconj,), (child,)],
                            passive=False,
                        )
                    )
                    break
                if strict_level == 1 and loosely_match(token, child):
                    self.logger.debug(
                        f"match conj {child.text!r} with strick level {strict_level}"
                    )
                    results.append(
                        MatchedPattern(
                            pattern=["V", "and", "v"],
                            tag=["Headword", "Grammar", "Content"],
                            token=[(token,), (cconj,), (child,)],
                            passive=False,
                        )
                    )
                    break
                if strict_level == 2 and child.tag_ == token.tag_:
                    self.logger.debug(
                        f"match conj {child.text!r} with strick level {strict_level}"
                    )
                    results.append(
                        MatchedPattern(
                            pattern=["V", "and", "v"],
                            tag=["Headword", "Grammar", "Content"],
                            token=[(token,), (cconj,), (child,)],
                            passive=False,
                        )
                    )
                    break

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_wh_to_inf(
        self, token: Token, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        if not is_verb(token):
            return

        pattern = "V wh-to-inf"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        """
        1. Find existing complement root
        2. Check if the root is a to-inf
        3. Check if the wh-word appears before the to-inf
        """

        # 1. find existing complement root #
        comp_roots, is_passive_voice = [], False
        if is_pp(token):
            self.logger.debug("checking for passive voice")
            comp_roots = []
            for child in token.lefts:
                if is_passive_be(child):
                    is_passive_voice = True
                    break

                if child.dep_ in ["csubjpass", "advcl"]:
                    comp_roots.append(child)

            if not is_passive_voice:
                self.logger.debug("No BE-verb is found. All passive roots are ignored.")
                comp_roots.clear()

        if not is_passive_voice:
            self.logger.debug("checking for active voice")
            for child in token.rights:
                if child.dep_ in ["xcomp", "advcl"]:
                    comp_roots.append(child)

        if len(comp_roots) == 0:
            self.logger.debug(f"complement for {token.text!r} not found. abort.")
            return

        self.logger.debug(f"found {len(comp_roots)} candidate roots: {comp_roots}")
        for root in comp_roots:
            # 2. check if the root is a to-inf #
            if not (to_inf := self.to_INF(root)):
                self.logger.debug(f"root {root.text!r} is not a to-inf. ignored.")
                continue

            to = to_inf[0][0]  # extract token from the pattern result

            # 3. check if there's a wh-word
            for child in root.lefts:
                if child.i >= to.i:
                    break
                if child.is_punct:
                    continue

                if self.pattern.is_wh_word(child.lemma_):
                    self.logger.debug("found wh-to-inf")
                    results.append(
                        MatchedPattern(
                            pattern=["V", "wh", "to", "inf"],
                            tag=["Headword", "Grammar", "Grammar", "Content"],
                            token=[(token,), (child,), (to,), (root,)],
                            passive=is_passive_voice,
                        )
                    )
                    break

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_wh(
        self, token: Token, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        if not is_verb(token):
            return

        pattern = "V wh"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        """
        1. Find complement roots
        2. Extract wh-clauses from roots
        """
        # 1. find existing complement root #
        comp_roots, is_passive_voice = [], False
        if is_pp(token):
            self.logger.debug("checking for passive voice")
            for child in token.lefts:
                if is_passive_be(child):
                    is_passive_voice = True
                    break

                if child.dep_ in ["csubjpass", "advcl"]:
                    comp_roots.append(child)

            if not is_passive_voice:
                self.logger.debug("No BE-verb is found. All passive roots are ignored.")
                comp_roots.clear()

        if not is_passive_voice:
            self.logger.debug("checking for active voice")
            for child in token.rights:
                if child.dep_ == "ccomp":
                    comp_roots.append(child)

        if len(comp_roots) == 0:
            self.logger.debug(f"complement for {token.text!r} not found. abort.")
            return

        self.logger.debug(f"found {len(comp_roots)} candidate roots: {comp_roots}")
        for root in comp_roots:
            # 2. extract wh-clauses
            wh_clauses = self.clause(root, skip_that=True, skip_comp=True)
            if not wh_clauses:
                self.logger.debug(f"wh-clauses not found. root {root.text!r} ignored.")
                continue

            for clause in wh_clauses:
                self.logger.debug("found wh-clause")
                results.append(
                    MatchedPattern(
                        pattern=["V", "wh"],
                        tag=["Headword", "Grammar"],
                        token=[(token,), clause.token[0]],
                        passive=is_passive_voice,
                    )
                )
                break

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_that(
        self, token: Token, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        """
        Note: not extracting passive voice yet.
        """
        if not is_verb(token):
            return

        pattern = "V that"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        """
        1. Find complement roots
        2. Extract clauses from roots
        3. Validate if it's a that-clause
        """
        # 1. find complement roots
        comp_roots = []
        for child in token.rights:
            if child.dep_ == "ccomp":
                comp_roots.append(child)

        if len(comp_roots) == 0:
            self.logger.debug(f"complement for {token.text!r} not found. abort")
            return

        self.logger.debug(f"found {len(comp_roots)} candidate roots: {comp_roots}")
        for root in comp_roots:
            # 2. extract clauses
            that_clauses = self.clause(root, skip_wh=True, skip_comp=True)
            if not that_clauses:
                self.logger.debug(
                    f"that-clauses not found. root {root.text!r} ignored."
                )
                continue

            # 3. validate it's a that-clause)
            for clause in that_clauses:
                if clause.pattern[0] == "that":
                    that = clause.token[0]
                else:
                    that = tuple()

                self.logger.debug(f"found a that-clause: {clause.pattern}")
                results.append(
                    MatchedPattern(
                        pattern=["V", "that"],
                        tag=["Headword", "Grammar"],
                        token=[(token,), that],
                        passive=False,
                    )
                )
                break

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_inf(
        self, token: Token, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        if not is_verb(token):
            return

        pattern = "V inf"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        for child in token.rights:
            if is_inf(child):
                self.logger.debug(f"found inf {child.text!r}")
                results.append(
                    MatchedPattern(
                        pattern=["V", "inf"],
                        tag=["Headword", "Content"],
                        token=[(token,), (child,)],
                        passive=False,
                    )
                )
                break

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_to_inf(
        self, token: Token, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        if not is_verb(token):
            return

        pattern = "V to-inf"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        for child in token.rights:
            if to_inf := self.to_INF(child):
                self.logger.debug(f"found to-inf {child.text!r}")
                to, inf = to_inf
                results.append(
                    MatchedPattern(
                        pattern=["V", "to", "inf"],
                        tag=["Headword", "Grammar", "Content"],
                        token=[(token,), to, inf],
                        passive=False,
                    )
                )
                break

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_ing(
        self, token: Token, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        if not is_verb(token):
            return

        pattern = "V -ing"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f'extracting pattern "V -ing" from token {token.text!r}')
        results = []
        """
        1. Check for passive voice
        2. find the active -ing group
           - I start (reading).
           - She tries to avoid (being caught).
        3. check aux V-ing
           - When you come shopping next time.
        """

        # 1. check for passive voice
        is_passive_voice = False
        if is_pp(token):
            self.logger.debug("checking for passive voice")
            for child in token.lefts:
                if child.is_punct:
                    continue

                if is_passive_be(child):
                    is_passive_voice = True
                    break

                if is_ing(child) and child.dep_ == "csubjpass":
                    results.append(
                        MatchedPattern(
                            pattern=["V", "-ing"],
                            tag=["Headword", "Grammar"],
                            token=[(token,), (child,)],
                            passive=True,
                        )
                    )

            if not is_passive_voice:
                self.logger.debug("No BE-verb is found. All passive -ing are ignored.")
                results.clear()

        # 2. find the active -ing group
        Ving_groups = []
        if not is_passive_voice:
            self.logger.debug("checking for active voice")
            for child in token.rights:
                # V v-ing
                if is_ing(child) and child.dep_ in ["xcomp", "dobj", "advcl"]:
                    self.logger.debug(f"match -ing {child.text!r}")
                    Ving_groups.append((child,))
                # V being V-ed
                elif is_pp(child):
                    for subchild in child.lefts:
                        if is_passive_be(subchild) and is_ing(subchild):
                            Ving_groups.append((subchild, child))

        if Ving_groups:
            for Ving in Ving_groups:
                results.append(
                    MatchedPattern(
                        pattern=["V", "-ing"],
                        tag=["Headword", "Grammar"],
                        token=[(token,), Ving],
                        passive=False,
                    )
                )
        else:
            # 3. check for `aux V-ing`
            head = token.head
            if is_ing(head) and token.dep_ == "aux":
                results.append(
                    MatchedPattern(
                        pattern=["V", "-ing"],
                        tag=["Headword", "Grammar"],
                        token=[(token,), (head,)],
                        passive=False,
                    )
                )

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_adj(
        self, token: Token, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        if not is_verb(token):
            return

        pattern = "V adj"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        is_passive_voice = False
        if is_pp(token):
            self.logger.debug("checking for passive voice")
            for child in token.lefts:
                if child.is_punct:
                    continue

                if is_passive_be(child):
                    is_passive_voice = True
                    break

        if is_passive_voice:
            self.logger.debug("This pattern is not used in passive voice. abort.")
            return

        for child in token.rights:
            if is_adj(child) and child.dep_ in ["acomp", "advcl", "advmod"]:
                self.logger.debug(f"match adjective {token.text!r}")
                results.append(
                    MatchedPattern(
                        pattern=["V", "adj"],
                        tag=["Headword", "Grammar"],
                        token=[(token,), (child,)],
                        passive=False,
                    )
                )
                break

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_pln(
        self, token: Token, check_pattern: bool = False
    ) -> Optional[List[MatchedPattern]]:
        """
        Only extracts the simplest form on page 59.
        All P and complex passive voice are not processed.
        """
        if not is_verb(token):
            return

        pattern = "V pl-n"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        is_passive_voice = False
        if is_pp(token):
            self.logger.debug("checking for passive voice")
            for child in token.lefts:
                if child.is_punct:
                    continue

                if is_passive_be(child):
                    is_passive_voice = True
                    break

                if is_pln(child) and child.dep_ == "nsubjpass":
                    self.logger.debug(f"found passive plural noun {child.text!r}")
                    results.append(
                        MatchedPattern(
                            pattern=["V", "pl-n"],
                            tag=["Headword", "Content"],
                            token=[(token,), (child,)],
                            passive=True,
                        )
                    )

            if not is_passive_voice:
                self.logger.debug("No BE-verb is found. All passive nouns are ignored.")
                results.clear()

        if not is_passive_voice:
            self.logger.debug("checking for active voice")
            for child in token.rights:
                if child.is_punct:
                    continue

                if is_pln(child) and child.dep_ in ["attr", "dobj", "iobj", "obj"]:
                    self.logger.debug(f"found plural noun {child.text!r}")
                    results.append(
                        MatchedPattern(
                            pattern=["V", "pl-n"],
                            tag=["Headword", "Content"],
                            token=[(token,), (child,)],
                            passive=False,
                        )
                    )

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def V_n(
        self, token: Token, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        """
        Only extracts the simplest form on page 14.
        All P and complex passive voice are not processed.
        """
        if not is_verb(token):
            return

        pattern = "V n"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        is_passive_voice = False
        if token.tag_ == "VBN":
            self.logger.debug("checking for passive voice")
            for child in token.lefts:
                if child.is_punct:
                    continue

                if is_passive_be(child):
                    is_passive_voice = True
                    break

                if is_noun(child) and child.dep_ == "nsubjpass":
                    self.logger.debug(f"found passive noun {child.text!r}")
                    results.append(
                        MatchedPattern(
                            pattern=["V", "n"],
                            tag=["Headword", "Content"],
                            token=[(token,), (child,)],
                            passive=True,
                        )
                    )

            if not is_passive_voice:
                self.logger.debug("No BE-verb is found. All passive nouns are ignored.")
                results.clear()

        if not is_passive_voice:
            self.logger.debug("checking for active voice")
            for child in token.rights:
                if child.is_punct:
                    continue

                if is_noun(child) and child.dep_ in ["attr", "dobj", "iobj", "obj"]:
                    self.logger.debug(f"found noun {child.text!r}")
                    results.append(
                        MatchedPattern(
                            pattern=["V", "n"],
                            tag=["Headword", "Content"],
                            token=[(token,), (child,)],
                            passive=False,
                        )
                    )

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    # ADJ #

    def ADJ_prep_n(
        self, token: Token, prep: str, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        """
        allow complement:
            about, against, at, for, from, in, of, on, over, to, toward, towards, with
        """
        if not is_adj(token):
            return

        pattern = f"ADJ {prep} n"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        # find preposition
        p = None
        for child in token.rights:
            if child.is_punct:
                continue

            if (
                child.lemma_ == prep
                and is_prep(child)
                or prep == "by"
                and child.dep_ == "agent"
            ):
                self.logger.debug("found preposition")
                p = child
                break

        if not p:
            head = token.head
            self.logger.debug(f"try to find preposition from head {head.text!r}")
            for child in head.rights:
                if child.is_punct:
                    continue

                if child.lemma_ == prep and is_prep(child):
                    p = child
                    break

        if not p:
            self.logger.debug(f"preposition {prep!r} not found. abort.")
            return

        # find n
        for child in p.rights:
            if child.is_punct:
                continue

            if child.dep_ == "pobj":
                self.logger.debug(f"found noun {child.text!r}")
                results.append(
                    MatchedPattern(
                        pattern=["ADJ", prep, "n"],
                        tag=["Headword", "Grammar", "Content"],
                        token=[(token,), (p,), (child,)],
                        passive=None,
                    )
                )
            elif prep in [
                "about",
                "at",
                "by",
                "for",
                "in",
                "of",
                "on",
                "upon",
                "over",
                "to",
                "toward",
                "towards",
                "with",
            ] and child.dep_ in ["pcomp"]:
                self.logger.debug(f"found noun complement root {child.text!r}")

                # check if it's a clause
                if comp_results := self.clause(child):
                    comp = comp_results[0]
                    headword = comp.token[comp.tag.index("Headword")][0]
                    if headword == child:
                        content = utility.unpack(comp.token)
                        results.append(
                            MatchedPattern(
                                pattern=["ADJ", prep, "n"],
                                tag=["Headword", "Grammar", "Content"],
                                token=[(token,), (p,), tuple(content)],
                                passive=None,
                            )
                        )
                        continue

                results.append(
                    MatchedPattern(
                        pattern=["ADJ", prep, "n"],
                        tag=["Headword", "Grammar", "Content"],
                        token=[(token,), (p,), (child,)],
                        passive=None,
                    )
                )

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    # NOUN #

    def N_prep_n(
        self, token: Token, prep: str, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        """
        allow complement: about, against, at, for, from, in, of, on, over, to, toward, towards, with
        """
        if not is_noun(token):
            return

        pattern = f"N {prep} n"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        # find preposition
        prep_word = None
        for child in token.rights:
            if child.is_punct:
                continue

            if child.lemma_ == prep and is_prep(child):
                prep_word = child
                break

        if not prep_word:
            self.logger.debug(f"preposition {prep!r} not found. abort.")
            return

        # find n
        for child in prep_word.rights:
            if child.is_punct:
                continue

            if child.dep_ == "pobj":
                self.logger.debug(f"found noun {child.text!r}")
                results.append(
                    MatchedPattern(
                        pattern=["N", prep, "n"],
                        tag=["Headword", "Grammar", "Content"],
                        token=[(token,), (prep_word,), (child,)],
                        passive=None,
                    )
                )
            elif (
                prep
                in [
                    "about",
                    "against",
                    "at",
                    "for",
                    "from",
                    "in",
                    "of",
                    "on",
                    "over",
                    "to",
                    "toward",
                    "towards",
                    "with",
                ]
                and child.dep_ in ["pcomp", "advcl"]
                and (comp := self.clause(child))
            ):
                self.logger.debug(f"found noun complement from root {child.text!r}")
                content = utility.unpack(comp[0].token)

                results.append(
                    MatchedPattern(
                        pattern=["N", prep, "n"],
                        tag=["Headword", "Grammar", "Content"],
                        token=[(token,), (prep_word,), tuple(content)],
                        passive=None,
                    )
                )

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def prep_N(
        self, token: Token, prep: str, check_pattern: bool = True
    ) -> Optional[List[MatchedPattern]]:
        """Checks for pattern prep + N.

        Args:
            token: the token to analyze from
            check_pattern:
                whether to check if the given pattern explicitly exists in Collins Grammar Pattern.
        """
        if not is_noun(token):
            return

        pattern = f"{prep} N"
        if check_pattern and not self.is_valid_pattern(token, pattern):
            return

        self.logger.debug(f"extracting pattern {pattern} from token {token.text!r}")
        results = []

        head = token.head
        if head.text == prep and token.dep_ == "pobj":
            results.append(
                MatchedPattern(
                    pattern=[prep, "N"],
                    tag=["Grammar", "Headword"],
                    token=[(head,), (token,)],
                    passive=None,
                )
            )

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    # ------ SEGMENTS ------ #

    def clause(
        self,
        root: Token,
        skip_wh: bool = False,
        skip_that: bool = False,
        skip_comp: bool = False,
    ) -> Optional[List[MatchedPattern]]:
        """Extracts any possibly existing clausal structure.

        Extractable clause:
          1. wh + to-inf
          2. wh + S + V
          3. [ that + ] S + V
          4. (if allow_comp) Ving / Ved
        """
        self.logger.debug(f"checking for clause pattern from token {root.text!r}")

        related_tokens = utility.sort_tokens(self._collect_descendant(root, level=0))
        self.logger.debug(f"get all related tokens {related_tokens}")

        if not skip_wh and (wh_clause := self.wh_clause(related_tokens)):
            self.logger.debug("match wh-clause")
            return wh_clause
        if not skip_that and (that_clause := self.that_clause(related_tokens)):
            self.logger.debug("match that-clause")
            return that_clause
        if not skip_comp and (comp := self.gerund_or_participle(related_tokens)):
            self.logger.debug("match gerund or participle")
            return comp
        return

    def that_clause(self, token_list: List[Token]) -> Optional[List[MatchedPattern]]:
        """[ that + ] S + V

        Gerund and participle are not included in this analysis.
        """
        self.logger.debug("try to find that-clause structure")
        results = []

        """
        1. find possible that token
        2. find verb
        3. find subject
        """
        that = None
        for token in token_list:
            # find that
            if token.text == "that" and token.tag_ == "IN":
                self.logger.debug(f"find that: {token.text!r}")
                that = token

            # find verb
            if not is_verb(token):
                continue
            self.logger.debug(f"find verb {token.text!r}")

            # find subject
            S = None
            passive = None
            for child in token.lefts:
                if child.is_punct:
                    continue

                if child.dep_ in ["nsubj", "csubj", "expl"]:
                    self.logger.debug(f"find active subject {child.text!r}")
                    S = child
                    passive = False
                    break
                elif child.dep_ in ["nsubjpass", "csubjpass"]:
                    self.logger.debug(f"find passive subject {child.text!r}")
                    S = child
                    passive = False
                    break

            if not S:
                self.logger.debug("No subject is found. Verb ignored.")
                continue

            if that:
                self.logger.debug("found a clause: that + S + V")
                results.append(
                    MatchedPattern(
                        pattern=["that", "S", "V"],
                        tag=["Grammar", "Content", "Headword"],
                        token=[(that,), (child,), (token,)],
                        passive=passive,
                    )
                )
            else:
                self.logger.debug("found a clause: S + V")
                results.append(
                    MatchedPattern(
                        pattern=["S", "V"],
                        tag=["Content", "Headword"],
                        token=[(child,), (token,)],
                        passive=passive,
                    )
                )
            break

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def wh_clause(self, token_list: List[Token]) -> Optional[List[MatchedPattern]]:
        """wh + S + V, or wh-to-inf"""
        self.logger.debug("try to find wh-clause structure")
        results = []

        """
        1. find wh-word
        2. find verb
           - check S + V
           - check to-inf
        """

        wh_word = None
        for token in token_list:
            # find wh-word
            if self.pattern.is_wh_word(token.text):
                self.logger.debug(f"find wh-word: {token.text!r}")
                wh_word = token
            if not wh_word:
                continue
            if not is_verb(token):
                continue

            self.logger.debug(f"checking verb {token.text!r}")

            # find subject
            S = None
            passive = None
            for child in token.lefts:
                if child.is_punct:
                    continue

                if child.dep_ in ["nsubj", "csubj"]:
                    self.logger.debug(f"find active subject {child.text!r}")
                    S = child
                    passive = False
                    break
                if child.dep_ in ["nsubjpass", "csubjpass"]:
                    self.logger.debug(f"find passive subject {child.text!r}")
                    S = child
                    passive = True
                    break

            if S:
                results.append(
                    MatchedPattern(
                        pattern=["wh", "S", "V"],
                        tag=["Grammar", "Content", "Headword"],
                        token=[(wh_word,), (S,), (token,)],
                        passive=passive,
                    )
                )
                break

            # find to-inf
            self.logger.debug("wh + S + V not found. checking wh-to-inf.")
            if to_inf := self.to_INF(token):
                self.logger.debug("match wh-to-inf")

                to = to_inf[0][0]  # extract token from the parse result
                results.append(
                    MatchedPattern(
                        pattern=["wh", "to", "inf"],
                        tag=["Grammar", "Grammar", "Headword"],
                        token=[(wh_word,), (to,), (token,)],
                        passive=None,
                    )
                )
                break

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def gerund_or_participle(
        self, token_list: List[Token]
    ) -> Optional[List[MatchedPattern]]:
        self.logger.debug("try to find gerund or paticiple structure")
        results = []

        for token in token_list:
            if not is_verb(token):
                continue

            if is_ing(token):
                results.append(
                    MatchedPattern(
                        pattern=["-ing"],
                        tag=["Headword"],
                        token=[(token,)],
                        passive=False,
                    )
                )
            elif is_pp(token):
                results.append(
                    MatchedPattern(
                        pattern=["-ed"],
                        tag=["Headword"],
                        token=[(token,)],
                        passive=True,
                    )
                )

            break  # gerunds or participles always appear as the beginning verb

        self.logger.debug(f"{len(results)} patterns are found.")
        return results

    def to_INF(self, token: Token) -> Optional[List[MatchedPattern]]:
        if token.tag_ not in ["VB", "VBN"]:
            return

        self.logger.debug(f'extracting pattern "to INF" from token {token.text!r}')
        """
        1. Extract inf
            (1) to V: to play, to sing, to type.
            (2) to Ved: to be played, to be written, etc.
        2. Extract 'to'
        """
        # 1. inf
        inf = None
        if token.tag_ == "VB":
            self.logger.debug("match inf as VB.")
            inf = (token,)
        elif token.tag_ == "VBN":
            for child in token.lefts:
                if is_passive_be(child):
                    self.logger.debug("match inf as be + VBN")
                    inf = (child, token)
        if not inf:
            self.logger.debug("no inf is found. abort.")
            return

        # 2. to
        for child in token.lefts:
            if child.text == "to" and child.tag_ == "TO" and child.dep_ == "aux":
                return [(child,), inf]
        return

    # ------ UTILITIES ------ #
    def is_valid_pattern(
        self, token: Token, pattern: str, strict: bool = False
    ) -> bool:
        if self.pattern.check(token.lemma_, pattern):
            return True
        if strict:
            return False
        return self.pattern.check(token.text.lower(), pattern)

    def _search_wh_word(self, root: Token) -> Optional[Token]:
        """Deeply search for the wh_word that precedes the root.

        Some wh-word may not have direct dependency with the clause root
          (e.g., "You didn't see how happy she *became*."),
          so the deep search is needed.
        Note that this only checks for the word, not checking the dependency tag for the wh-word,
          since the valid tags depend on the pattern.
        """
        if self.pattern.is_wh_word(root.lemma_):
            return root
        for child in root.children:
            search = self._search_wh_word(child)
            if search:
                return search
        return

    def _collect_descendant(
        self, root: Token, level: int, exclude_punct: bool = False
    ) -> Set[Token]:
        """Recursively collects all descendants from the given root.

        Args:
            root: The token to start collecting from,
            level: the filter level during the collecting.
              0: all descendants, regardless of dependencies.
              1: ignore coordinate conjunction and appositions.
              2: ignore clausal modifiers and complements.
              3: ignore all modifiers.
        """
        # level 0
        children = root.children
        if level >= 1:
            children = filter(
                lambda child: child.dep_ not in ["appos", "cc", "conj", "preconj"],
                children,
            )
        if level >= 2:
            children = filter(
                lambda child: child.dep_
                not in ["acl", "advcl", "ccomp", "rcmod", "relcl", "xcomp"],
                children,
            )
        if level >= 3:
            children = filter(
                lambda child: child.dep_
                not in [
                    "appos",
                    "amod",
                    "advmod",
                    "infmod",
                    "nmod",
                    "nn",
                    "npadvmod",
                    "npmod",
                    "prep",
                    "meta",
                    "neg",
                ],  # not sure
                children,
            )
        if level >= 4:
            self.logger.warn(
                f"{PatternValidator._collect_descendant.__name__} - "
                f"Unknown level {level}. Request ignored."
            )

        if exclude_punct:
            children = filter(lambda child: not child.is_punct, children)

        children = [*children]
        results = set([root, *children])
        for child in children:
            results.update(self._collect_descendant(child, level=level))

        return results
