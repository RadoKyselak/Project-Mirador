from typing import Dict

from .formula import Predicate
from .terms import Constant, Term, Variable, as_term

Substitution = Dict[str, Term]


def apply_substitution(term: Term, substitution: Substitution) -> Term:
    current = term
    while isinstance(current, Variable) and current.name in substitution:
        next_term = substitution[current.name]
        if next_term == current:
            break
        current = next_term
    return current


def _occurs_check(variable: Variable, term: Term, substitution: Substitution) -> bool:
    resolved = apply_substitution(term, substitution)
    return isinstance(resolved, Variable) and resolved.name == variable.name


def _bind(variable: Variable, term: Term, substitution: Substitution) -> Substitution | None:
    if _occurs_check(variable, term, substitution):
        return None
    updated = dict(substitution)
    updated[variable.name] = term
    return updated


def unify_terms(left: Term, right: Term, substitution: Substitution) -> Substitution | None:
    left_resolved = apply_substitution(left, substitution)
    right_resolved = apply_substitution(right, substitution)

    if left_resolved == right_resolved:
        return substitution
    if isinstance(left_resolved, Variable):
        return _bind(left_resolved, right_resolved, substitution)
    if isinstance(right_resolved, Variable):
        return _bind(right_resolved, left_resolved, substitution)
    if isinstance(left_resolved, Constant) and isinstance(right_resolved, Constant):
        return substitution if left_resolved.value == right_resolved.value else None
    return None


def unify(left: Predicate, right: Predicate, substitution: Substitution | None = None) -> Substitution | None:
    if left.name != right.name or len(left.args) != len(right.args):
        return None

    current_substitution = substitution or {}
    for left_arg, right_arg in zip(left.args, right.args):
        result = unify_terms(as_term(left_arg), as_term(right_arg), current_substitution)
        if result is None:
            return None
        current_substitution = result
    return current_substitution

