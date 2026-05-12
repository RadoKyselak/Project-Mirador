from __future__ import annotations

from collections import defaultdict
from typing import Generator

from .formula import Comparison, Predicate, Rule
from .terms import Constant, Variable
from .unification import Substitution, apply_substitution, unify


class KnowledgeBase:
    def __init__(self) -> None:
        self._facts: dict[str, list[Predicate]] = defaultdict(list)
        self._rules: list[Rule] = []
        self._rule_counter = 0

    def add_fact(self, fact: Predicate) -> None:
        self._facts[fact.name].append(fact)

    def add_rule(self, rule: Rule) -> None:
        self._rules.append(rule)

    def ask(self, query: Predicate, max_depth: int = 12) -> bool:
        return any(self.query(query, max_depth=max_depth))

    def query(self, query: Predicate, max_depth: int = 12) -> Generator[Substitution, None, None]:
        yield from self._solve([query], {}, max_depth)

    def _solve(self, goals: list[Predicate | Comparison], substitution: Substitution, depth: int) -> Generator[Substitution, None, None]:
        if depth < 0:
            return
        if not goals:
            yield substitution
            return

        current_goal = self._substitute_goal(goals[0], substitution)
        remaining_goals = goals[1:]

        if isinstance(current_goal, Comparison):
            if self._evaluate_comparison(current_goal):
                yield from self._solve(remaining_goals, substitution, depth)
            return

        for fact in self._facts.get(current_goal.name, []):
            fact_substitution = unify(current_goal, fact, substitution)
            if fact_substitution is not None:
                yield from self._solve(remaining_goals, fact_substitution, depth - 1)

        for rule in self._rules:
            if rule.consequent.name != current_goal.name:
                continue
            rule_instance = self._standardize_rule(rule)
            rule_substitution = unify(current_goal, rule_instance.consequent, substitution)
            if rule_substitution is None:
                continue
            next_goals = list(rule_instance.antecedents) + remaining_goals
            yield from self._solve(next_goals, rule_substitution, depth - 1)

    def _standardize_rule(self, rule: Rule) -> Rule:
        self._rule_counter += 1
        suffix = f"_{self._rule_counter}"
        rename_map = {var_name: Variable(f"{var_name}{suffix}") for var_name in rule.variables}

        def rename_term(term: Constant | Variable):
            if isinstance(term, Variable):
                return rename_map.get(term.name, term)
            return term

        antecedents: list[Predicate | Comparison] = []
        for antecedent in rule.antecedents:
            if isinstance(antecedent, Predicate):
                antecedents.append(Predicate(antecedent.name, tuple(rename_term(arg) for arg in antecedent.args)))
            else:
                antecedents.append(
                    Comparison(
                        antecedent.operator,
                        rename_term(antecedent.left),
                        rename_term(antecedent.right),
                    )
                )

        consequent = Predicate(rule.consequent.name, tuple(rename_term(arg) for arg in rule.consequent.args))
        return Rule(variables=tuple(rename_map.keys()), antecedents=tuple(antecedents), consequent=consequent)

    def _substitute_goal(self, goal: Predicate | Comparison, substitution: Substitution) -> Predicate | Comparison:
        if isinstance(goal, Predicate):
            return Predicate(goal.name, tuple(apply_substitution(arg, substitution) for arg in goal.args))
        return Comparison(
            goal.operator,
            apply_substitution(goal.left, substitution),
            apply_substitution(goal.right, substitution),
        )

    @staticmethod
    def _evaluate_comparison(comparison: Comparison) -> bool:
        left = comparison.left
        right = comparison.right
        if isinstance(left, Variable) or isinstance(right, Variable):
            return False

        left_value = left.value
        right_value = right.value
        op = comparison.operator
        if op == ">":
            return left_value > right_value
        if op == "<":
            return left_value < right_value
        if op == ">=":
            return left_value >= right_value
        if op == "<=":
            return left_value <= right_value
        if op == "==":
            return left_value == right_value
        if op == "!=":
            return left_value != right_value
        return False

