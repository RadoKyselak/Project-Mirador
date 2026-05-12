from dataclasses import dataclass
from typing import Literal

from .terms import Term, as_term


@dataclass(frozen=True)
class Predicate:
    name: str
    args: tuple[Term, ...]

    def __init__(self, name: str, args: tuple[Term | object, ...] | list[Term | object]):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "args", tuple(as_term(arg) for arg in args))


@dataclass(frozen=True)
class Comparison:
    operator: Literal[">", "<", ">=", "<=", "==", "!="]
    left: Term
    right: Term

    def __init__(self, operator: Literal[">", "<", ">=", "<=", "==", "!="], left: Term | object, right: Term | object):
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "left", as_term(left))
        object.__setattr__(self, "right", as_term(right))


@dataclass(frozen=True)
class And:
    clauses: tuple[Predicate | Comparison, ...]


@dataclass(frozen=True)
class Or:
    clauses: tuple[Predicate | Comparison, ...]


@dataclass(frozen=True)
class Not:
    clause: Predicate | Comparison


@dataclass(frozen=True)
class Implies:
    premise: Predicate | Comparison | And
    conclusion: Predicate


@dataclass(frozen=True)
class ForAll:
    variables: tuple[str, ...]
    formula: Implies


@dataclass(frozen=True)
class Exists:
    variables: tuple[str, ...]
    formula: Predicate


@dataclass(frozen=True)
class Rule:
    variables: tuple[str, ...]
    antecedents: tuple[Predicate | Comparison, ...]
    consequent: Predicate

