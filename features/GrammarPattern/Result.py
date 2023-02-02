from dataclasses import dataclass
from typing import List, Tuple

from spacy.tokens import Token
from typing_extensions import Self

TokenPack = Tuple[Token]
IndexPack = Tuple[int]


@dataclass(frozen=True)
class MatchedPattern:
    pattern: List[str]
    tag: List[str]
    token: List[TokenPack]
    passive: bool

    def __repr__(self):
        output = []
        output.append("{")
        for k, v in self.__dict__.items():
            output.append(f"  '{k}': {v!r}")
        output.append("}")
        return "\n".join(output)

    def __getitem__(self, key):
        return self.__getattribute__(key)


@dataclass(frozen=True)
class PatternResult:
    pattern: List[str]
    passive: bool
    tag: List[str]
    pattern_tag: List[str]
    token: List[TokenPack]
    token_index: List[IndexPack]
    chunk: List[TokenPack]
    chunk_index: List[IndexPack]

    def __repr__(self):
        output = []
        output.append("{")
        for k, v in self.__dict__.items():
            output.append(f"  {k!r}: {v!r}")
        output.append("}")
        return "\n".join(output)

    def __getitem__(self, key):
        return self.__getattribute__(key)


@dataclass(frozen=True)
class SimplePatternResult:
    pattern: str
    passive: bool
    tag: str
    token: str
    chunk: str

    def __repr__(self):
        output = []
        output.append("{")
        for k, v in self.__dict__.items():
            output.append(f"  {k!r}: {v!r}")
        output.append("}")
        return "\n".join(output)

    def __getitem__(self, key):
        return self.__getattribute__(key)


@dataclass(frozen=True)
class TokenResult:
    token: Token
    pattern: List[PatternResult]

    def update(self, new_results: Self) -> Self:
        assert new_results.token == self.token

        self.pattern.extend(new_results.patterns)
        return self

    def __repr__(self) -> str:
        return f"{{'token': {self.token.text!r}, pattern: {self.pattern}}}"

    def __getitem__(self, key):
        return self.__getattribute__(key)
