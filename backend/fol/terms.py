from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Variable:
    name: str


@dataclass(frozen=True)
class Constant:
    value: Any


Term = Variable | Constant


def as_term(value: Any) -> Term:
    if isinstance(value, (Variable, Constant)):
        return value
    return Constant(value)
