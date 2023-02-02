import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from spacy.tokens.token import Token

from .chunk_extractor import extract_chunk
from .PatternValidator import PatternValidator
from .Result import (
    MatchedPattern,
    PatternResult,
    SimplePatternResult,
    TokenPack,
    TokenResult,
)
from .token_checker import is_adj, is_noun, is_verb
from .utility import sort_tokens

WORKING_DIR = os.path.dirname(__file__)


class PatternParse:
    _DEFAULT_SPACY_MODEL = "en_core_web_lg"

    def __init__(
        self,
        sentence,
        target=None,
        model=None,
        pattern=None,
        longest=False,
        logger=None,
    ):
        """
        Args:
            sentence: The English sentence to parse. Can be plain text or processed spacy Doc.
            [target]: If given,
                parse only the patterns related to the target word,
                or all adjs, nouns and verbs will be parsed.
                Can be a spacy token, a token index, a string, or a iterator.
                If string is given, only the first match will be processed.
                Default `None`.
            [model]: The spacy model to use in the processing.
                Can be the pre-loaded model or a model name. Default `en_core_web_lg`.
            [longest]: (default False) whether to return longest matched pattern only.
            [pattern]: Pre-loaded `PatternHolder` object.
            [logger]: The logger used to log debug messages. Root logger is used by default.
        """
        # process string to spacy Doc
        if type(sentence) == str:
            sentence = re.sub(r"\s+", " ", sentence)
            model = self.load_spacy_model(model)
            self.doc = model(sentence)
        else:
            self.doc = sentence

        if target is None:
            # process all adjs, nouns, and verbs by default
            targets = filter(
                lambda t: t.pos_ in ["ADJ", "AUX", "NOUN", "PRON", "PROPN", "VERB"],
                self.doc,
            )
        elif type(target) == int:
            targets = [next(token for token in self.doc if token.i == target)]
        elif type(target) == str:
            targets = [
                next(
                    token
                    for token in self.doc
                    if token.text == target or token.lemma_ == target
                )
            ]
        elif type(target) == list:
            targets = target
        else:
            targets = [target]

        self.longest = longest
        self.logger = logger or logging.getLogger()
        self.pattern = PatternValidator(pattern=pattern)
        self.results = {}

        for token in targets:
            self.parse(token)

    # ------ INTERFACE ------ #
    def parse(self, token: Token) -> Optional[List[Dict]]:
        self.logger.info(f"Start to find patterns related to token {token.text!r}")
        results = self.parse_pattern_from_token(token)
        if not results:
            self.logger.info("pattern not found. abort.")
            return

        self.logger.info("Start to extracted word chunks")
        chunk_results = self.collect_word_chunk(results)

        self.logger.info("Start to postprocess the result")
        results = self.post_process(token, results, chunk_results)

        self.logger.info("Start to merge results")
        self.update_result(results)
        return results

    def simplify(self, print_result: bool = False):
        simple_results = defaultdict(list)
        for word_patterns in self.results.values():
            simple_patterns = []
            for pattern in word_patterns.pattern:
                token = self.merge_result_set(pattern.token)
                token = sort_tokens(token)
                chunk = self.merge_result_set(pattern.chunk)
                chunk = sort_tokens(chunk)
                simple_patterns.append(
                    SimplePatternResult(
                        pattern=self.flatten_to_str(pattern.pattern),
                        tag=self.flatten_to_str(pattern.pattern_tag),
                        token=self.flatten_to_str(token),
                        chunk=self.flatten_to_str(chunk),
                        passive=pattern.passive,
                    )
                )
            simple_results[word_patterns.token.text].extend(simple_patterns)

        simple_results = dict(simple_results)
        if print_result:
            print(simple_results)

        return simple_results

    # ------ PROCEDURE ------ #

    # step 1 #
    def parse_pattern_from_token(self, token: Token) -> List[MatchedPattern]:
        def check_early_return():
            return self.longest and len(patterns)

        patterns = []

        # Note that the pattern order MATTERS if early-return mode is on.

        if is_verb(token):
            self.logger.debug(f"checking patterns for verb {token.text!r}")

            # CHAPTER 4 ##
            preps = [
                "about",
                "against",
                "as",
                "at",
                "by",
                "for",
                "from",
                "in",
                "into",
                "of",
                "off",
                "on",
                "onto",
                "over",
                "to",
                "toward",
                "towards",
                "with",
            ]
            for prep in preps:
                if result := self.pattern.V_n_prep_n(token, prep=prep):
                    self.logger.debug(f"match pattern: V n {prep} n")
                    patterns.extend(result)

            if check_early_return():
                return patterns

            if result := self.pattern.V_n_prep(token):
                self.logger.debug("match pattern: V n prep")
                patterns.extend(result)
            if result := self.pattern.V_n_adv(token):
                self.logger.debug("match pattern: V n adv")
                patterns.extend(result)

            if check_early_return():
                return patterns

            # CHAPTER 3 #
            if result := self.pattern.V_n_that(token):
                self.logger.debug("match pattern: V n that")
                patterns.extend(result)
            if result := self.pattern.V_n_ing(token):
                self.logger.debug("match pattern: V n ing")
                patterns.extend(result)
            if result := self.pattern.V_n_n(token):
                self.logger.debug("match pattern: V n n")
                patterns.extend(result)

            if check_early_return():
                return patterns

            # CHAPTER 2 #
            preps = [
                "with",
                "under",
                "towards",
                "toward",
                "to",
                "through",
                "over",
                "onto",
                "on",
                "off",
                "of",
                "like",
                "into",
                "in",
                "from",
                "for",
                "at",
                "as",
                "round",
                "around",
                "against",
                "after",
                "across",
                "about",
            ]
            for prep in preps:
                if result := self.pattern.V_prep_n(token, prep=prep):
                    self.logger.debug(f"match pattern: V {prep} n")
                    patterns.extend(result)

            if check_early_return():
                return patterns

            if result := self.pattern.V_prep(token):
                self.logger.debug("match pattern: V prep")
                patterns.extend(result)
            if result := self.pattern.V_adv(token):
                self.logger.debug("match pattern: V adv")
                patterns.extend(result)

            if check_early_return():
                return patterns

            # CHAPTER 1 #
            if result := self.pattern.V_and_v(token):
                self.logger.debug("match pattern: V and v")
                patterns.extend(result)
            if result := self.pattern.V_wh_to_inf(token):
                self.logger.debug("match pattern: V wh-to-inf")
                patterns.extend(result)
            if result := self.pattern.V_wh(token):
                self.logger.debug("match pattern: V wh")
                patterns.extend(result)
            if result := self.pattern.V_that(token):
                self.logger.debug("match pattern: V that")
                patterns.extend(result)
            if result := self.pattern.V_inf(token):
                self.logger.debug("match pattern: V inf")
                patterns.extend(result)
            if result := self.pattern.V_to_inf(token):
                self.logger.debug("match pattern: V to inf")
                patterns.extend(result)
            if result := self.pattern.V_ing(token):
                self.logger.debug("match pattern: V -ing")
                patterns.extend(result)
            if result := self.pattern.V_adj(token):
                self.logger.debug("match pattern: V adj")
                patterns.extend(result)
            # if result := self.pattern.V_pln(token):
            #     self.logger.debug("match pattern: V pl-n")
            #     patterns.extend(result)

            if check_early_return():
                return patterns

            if result := self.pattern.V_n(token):
                self.logger.debug("match pattern: V n")
                patterns.extend(result)

        elif is_adj(token):
            self.logger.debug(f"checking patterns for adjective {token.text!r}")

            preps = [
                "about",
                "against",
                "as",
                "at",
                "by",
                "for",
                "from",
                "in",
                "of",
                "on",
                "upon",
                "over",
                "to",
                "toward",
                "towards",
                "with",
            ]
            for prep in preps:
                if result := self.pattern.ADJ_prep_n(
                    token, prep=prep, check_pattern=True
                ):
                    self.logger.debug(f"match pattern: ADJ {prep} n")
                    patterns.extend(result)

        elif is_noun(token):
            self.logger.debug(f"checking patterns for noun {token.text!r}")

            preps = [
                "about",
                "against",
                "as",
                "at",
                "behind",
                "by",
                "for",
                "from",
                "in",
                "into",
                "on",
                "over",
                "to",
                "toward",
                "towards",
                "with",
            ]
            for prep in preps:
                if result := self.pattern.N_prep_n(
                    token, prep=prep, check_pattern=True
                ):
                    self.logger.debug(f"match pattern: N {prep} n")
                    patterns.extend(result)
            preps = ["of"]
            for prep in preps:
                if result := self.pattern.N_prep_n(
                    token, prep=prep, check_pattern=False
                ):
                    self.logger.debug(f"match pattern: N {prep} n")
                    patterns.extend(result)

            if check_early_return():
                return patterns

            preps = [
                "at",
                "by",
                "from",
                "in",
                "into",
                "of",
                "on",
                "to",
                "under",
                "with",
                "within",
                "without",
            ]
            for prep in preps:
                if result := self.pattern.prep_N(token, prep=prep):
                    self.logger.debug(f"match pattern: {prep} n")
                    patterns.extend(result)

        return patterns

    # step 2 #
    def collect_word_chunk(
        self, results: List[MatchedPattern]
    ) -> List[List[TokenPack]]:
        """Extract the semetic chunk for the result tokens, and add them back into the dictionary.

        For example, give a sentence "The kid accidentally drops that cup of water.".
        Given a core token "kid", the noun chunk "The kid" will be extracted.
        Given a core token "drop", the verb chunk "accidentally drops" will be extracted.
        """
        chunk_results = []
        for result in results:
            word_chunks = []  # the result of the chunk extraction
            core_tokens = self.merge_result_set(result.token)
            for cur_core_tokens, grammar_tag in zip(result.token, result.tag):
                self.logger.debug(
                    f"extracting word chunk for tokens {cur_core_tokens!r}"
                )
                related_tokens = set()
                for token in cur_core_tokens:
                    chunk = extract_chunk(token, grammar_tag)
                    related_tokens.update(chunk)
                related_tokens = related_tokens - core_tokens.difference(
                    cur_core_tokens
                )  # core tokens should remain at original position
                word_chunks.append(related_tokens)
                core_tokens.update(cur_core_tokens)
            # transform into tuple
            word_chunks = [tuple(sort_tokens(chunks)) for chunks in word_chunks]
            self.logger.debug(f"get word chunks: {word_chunks}")
            chunk_results.append(word_chunks)
        return chunk_results

    # step 3 #
    def post_process(
        self,
        token: Token,
        results: List[MatchedPattern],
        chunk_results: List[TokenPack],
    ) -> Dict[int, TokenResult]:

        pattern_list = []
        for result, chunk in zip(results, chunk_results):
            pattern = result.pattern
            tags = self.process_tag(result.tag)
            pattern_tags = self.process_pattern_tag(pattern, tags)
            matched_tokens = result.token
            pattern_list.append(
                PatternResult(
                    pattern=pattern,
                    passive=result.passive,
                    tag=tags,
                    pattern_tag=pattern_tags,
                    token=matched_tokens,
                    token_index=self.extract_token_index(matched_tokens),
                    chunk=chunk,
                    chunk_index=self.extract_token_index(chunk),
                )
            )

        new_results = {
            token.i: TokenResult(
                token=token,
                pattern=pattern_list,
            )
        }
        return new_results

    # step 4 #
    def update_result(self, new_results: Dict[int, TokenResult]) -> None:
        for idx, data in new_results.items():
            if idx not in self.results:
                self.results[idx] = data
                return
            self.results[idx].update(data)

    # ------ UTILITIES ------ #

    def load_spacy_model(self, model=None):
        from spacy import load as load_model

        if model is None:
            return load_model(
                self._DEFAULT_SPACY_MODEL, exclude=["ner", "custom", "textcat"]
            )
        if type(model) == str:
            return load_model(model, exclude=["ner", "custom", "textcat"])
        return model

    def merge_result_set(self, token_list: List[Tuple]) -> set:
        """Merge the list of tuples into a set"""
        return set().union(*token_list)

    # ------ POST-PROCESSING ------ #

    @staticmethod
    def process_pattern_tag(pattern: List[str], tags: List[str]) -> List[str]:
        pattern_tag = pattern.copy()
        for i, tag in enumerate(tags):
            if "Content" in tag:
                pattern_tag[i] = tag
        return pattern_tag

    @staticmethod
    def process_tag(original_tag: List[str]) -> List[str]:
        """append indice after content tag (Content1, Content2, etc.)"""
        new_tag = original_tag[:]
        content_count = 0
        for i, tag in enumerate(new_tag):
            if tag == "Content":
                content_count += 1
                new_tag[i] = f"Content{content_count}"
        return new_tag

    @staticmethod
    def process_pattern_words(pattern_words: List[Tuple[str]]) -> List[Tuple[str]]:
        """Transform spacy.token into string"""
        return [tuple(token.text for token in seg) for seg in pattern_words]

    @staticmethod
    def extract_token_index(pattern_words: List[Tuple[str]]) -> List[Tuple[str]]:
        """Transform spacy.token into token index"""
        return [tuple(token.i for token in seg) for seg in pattern_words]

    @staticmethod
    def flatten_to_str(data_list: List, sep: str = " ") -> str:
        res = []
        for segment in data_list:
            if type(segment) == str:
                res.append(segment)
            elif type(segment) == Token:
                res.append(segment.text)
            else:
                res.extend(segment)
        res = map(lambda t: t.text if type(t) == Token else t, res)
        return sep.join(res)
