import json
import logging
import os
from itertools import chain, zip_longest
from typing import Iterable, List, Optional

WORKING_DIR = os.path.dirname(__file__)


class PatternHolder(object):
    _DEFAULT_DATA_FOLDER = os.path.join(WORKING_DIR, "data")

    def __init__(self, data_dir: str = None, logger: logging.Logger = None) -> None:
        self.data_dir = data_dir or PatternHolder._DEFAULT_DATA_FOLDER
        self._preposition_path = os.path.join(self.data_dir, "prepositions.txt")
        self._wh_word_path = os.path.join(self.data_dir, "wh_words.txt")
        self._collins_pattern_path = os.path.join(
            self.data_dir, "collins_grammar_patterns.json"
        )
        self.logger = logger if logger else logging.getLogger()

        # prepare the data
        self._load_data()

    # ------ INTERFACE ------ #

    def search(self, word: str, pos: Optional[str] = None) -> List[str]:
        """Searches all patterns that matches the word and pos.

        Args:
            word: the word of the patterns.
            pos: If given, only those patterns under the same pos will be returned,
              or all patterns about the word will be checked by default.
        """
        if pos:
            # TODO: Double-check
            # try:
            #     return self.collins_pattern[pos][word]
            # except:
            #     return []
            return self.collins_pattern[pos][word]

        all_patterns = [
            pos_data[word]
            for pos_data in self.collins_pattern.values()
            if word in pos_data
        ]
        return list(chain(*all_patterns))

    def check(
        self, word: str, pattern: str, pos: Optional[str] = None, strict: bool = True
    ) -> bool:
        """Check if the pattern is valid for the given word, under the optionally-given pos.

        Args:
            word: the root of the pattern.
            pattern: the pattern to check.
            pos: If given, only those patterns under the same pos will be returned,
              or all patterns about the word will be checked by default.
            strict: to strictly match the given pattern.
        """
        self.logger.debug(
            f"checking pattern {pattern!r} for word '{word}[{pos or '-'}]'"
        )

        exist_patterns = self.search(word, pos=pos)
        if pattern in exist_patterns:
            self.logger.debug("match under strick mode")
            return True
        if strict:
            return False

        self.logger.debug("try to find pattern under loose mode")
        target_pattern_words = pattern.split()
        for exist_pattern in exist_patterns:
            self.logger.debug(f"checking pattern {exist_pattern!r}")
            exist_pattern_words = exist_pattern.split()

            # loosely check target words
            loose_target_words = [self.stag(tag) for tag in target_pattern_words]
            match = self._loose_pattern_match(
                exist_pattern_words, (target_pattern_words, loose_target_words)
            )
            if match:
                match = " ".join(match)
                self.logger.debug(f"match with loose target: {pattern!r} -> {match!r}")
                return True

            # loosely check pattern words
            loose_existing_words = [self.stag(tag) for tag in exist_pattern_words]
            match = self._loose_pattern_match(
                target_pattern_words, (exist_pattern_words, loose_existing_words)
            )
            if match:
                match = " ".join(match)
                self.logger.debug(
                    f"match with loose pattern: {exist_pattern!r} -> {match!r}"
                )
                return True
        return False

    def stag(self, tag):
        """Simplifies the Collins pos tag.

        For example, [v, -ing, -ed, inf] are all mapped into 'v',
          and most preposition words are mapped into 'prep'.
        """

        def format(_tag):
            if tag.istitle():
                return _tag.capitalize()
            if tag.isupper():
                return _tag.upper()
            return _tag

        _tag = tag.lower()
        if _tag in self.preposition:
            return format("prep")
        if _tag in ["n", "pl-n", "pron-refl"]:
            return format("n")
        if _tag in ["v", "-ing", "-ed", "inf"]:
            return format("v")
        return tag

    def is_wh_word(self, word: str, use_extended: bool = True) -> bool:
        """Check if the given word is a wh-word.

        Args:
            word: the word to check
            use_extended:
              Set to `True` to accept wh-like words, even if it's not in the given wh-word list.
        """
        word = word.lower()
        if word in self.wh_word:
            return True
        if use_extended and word.startswith("wh"):
            return True
        return False

    # ------ UTILITIES ------ #

    def _load_data(self):
        self.logger.debug(
            f"loading Collins Grammar Pattern from {self._collins_pattern_path}"
        )
        with open(self._collins_pattern_path, "r") as f:
            data = json.load(f)
        self.collins_pattern = data

        self.logger.debug(f"loading Collins prepositions from {self._preposition_path}")
        with open(self._preposition_path) as f:
            data = f.read().split("\n")
        self.preposition = set(
            filter(lambda l: l.strip() and not l.startswith("#"), data)
        )

        self.logger.debug(f"loading Collins wh-words from {self._wh_word_path}")
        with open(self._wh_word_path) as f:
            data = f.read().split("\n")
        self.wh_word = set(filter(lambda l: l.strip() and not l.startswith("#"), data))

    def _loose_pattern_match(
        self, target: List, loose_group: Iterable
    ) -> Optional[List]:
        self.logger.debug(f"loosely checking {target!r} and {loose_group!r}")
        matched = []
        for tword, *gwords in zip_longest(target, *loose_group):
            has_match = False
            for gword in gwords:
                if tword == gword:
                    matched.append(gword)
                    has_match = True
                    break
            if not has_match:
                matched = None
                break
        return matched
