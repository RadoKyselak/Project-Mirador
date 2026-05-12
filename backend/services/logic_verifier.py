import re
from typing import Any, Dict, List, Tuple

from fol import Comparison, Constant, KnowledgeBase, Predicate, Rule, Variable


def fol_reason_about_claim(
    claim: str,
    analysis: Dict[str, Any],
    sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if analysis.get("claim_type") != "quantitative_comparison":
        return {"verdict": "Inconclusive", "summary": "", "evidence_links": []}

    kb = KnowledgeBase()
    fact_index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    categories = set()

    for source in sources:
        fact_data = _spending_fact_from_source(source)
        if not fact_data:
            continue
        category, year, amount, unit = fact_data
        kb.add_fact(Predicate("Spending", [Constant(category), Constant(year), Constant(amount), Constant(unit)]))
        fact_index[(category, year)] = {
            "amount": amount,
            "url": source.get("url", ""),
            "line_description": source.get("line_description", category),
        }
        categories.add(category)

    if not fact_index:
        return {"verdict": "Inconclusive", "summary": "", "evidence_links": []}

    _add_spending_rule(kb)

    category_a, category_b, year = _extract_claim_targets(claim, analysis, categories)
    if not category_a or not category_b:
        return {"verdict": "Inconclusive", "summary": "", "evidence_links": []}

    relation = (analysis.get("relationship") or "").lower()
    if not relation:
        relation = "greater than" if "more than" in claim.lower() or "greater" in claim.lower() else "less than"

    if "less" in relation or "<" in relation:
        category_a, category_b = category_b, category_a

    query = _build_query(category_a, category_b, year)
    opposite_query = _build_query(category_b, category_a, year)

    if kb.ask(query):
        return {
            "verdict": "Supported",
            "summary": _build_summary("Supported", category_a, category_b, year, fact_index),
            "evidence_links": _build_links(category_a, category_b, year, fact_index),
        }
    if kb.ask(opposite_query):
        return {
            "verdict": "Contradicted",
            "summary": _build_summary("Contradicted", category_a, category_b, year, fact_index),
            "evidence_links": _build_links(category_a, category_b, year, fact_index),
        }
    return {"verdict": "Inconclusive", "summary": "", "evidence_links": []}


def _add_spending_rule(kb: KnowledgeBase) -> None:
    p = Variable("p")
    q = Variable("q")
    y = Variable("y")
    a = Variable("a")
    b = Variable("b")
    u = Variable("u")

    kb.add_rule(
        Rule(
            variables=("p", "q", "y", "a", "b", "u"),
            antecedents=(
                Predicate("Spending", [p, y, a, u]),
                Predicate("Spending", [q, y, b, u]),
                Comparison(">", a, b),
            ),
            consequent=Predicate("GreaterSpending", [p, q, y]),
        )
    )


def _build_query(category_a: str, category_b: str, year: int | None) -> Predicate:
    if year is None:
        return Predicate("GreaterSpending", [Constant(category_a), Constant(category_b), Variable("yq")])
    return Predicate("GreaterSpending", [Constant(category_a), Constant(category_b), Constant(year)])


def _extract_claim_targets(claim: str, analysis: Dict[str, Any], categories: set[str]) -> Tuple[str | None, str | None, int | None]:
    text = " ".join([claim] + [str(e) for e in analysis.get("entities", [])]).lower()
    category_hits = sorted(
        [cat for cat in categories if cat in text],
        key=lambda c: text.find(c),
    )
    if len(category_hits) < 2:
        return None, None, _extract_year(text)
    return category_hits[0], category_hits[1], _extract_year(text)


def _extract_year(text: str) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return int(match.group(0)) if match else None


def _spending_fact_from_source(source: Dict[str, Any]) -> Tuple[str, int, float, str] | None:
    url = (source.get("url") or "").lower()
    if "apps.bea.gov" not in url:
        return None

    amount = source.get("data_value")
    if amount is None:
        return None

    year_raw = str(source.get("raw_year") or "")
    year_match = re.search(r"\b(19|20)\d{2}\b", year_raw)
    if not year_match:
        snippet = source.get("snippet") or ""
        year_match = re.search(r"\b(19|20)\d{2}\b", snippet)
    if not year_match:
        return None

    line_description = (source.get("line_description") or "").lower()
    if "defense" in line_description:
        category = "defense"
    elif "education" in line_description:
        category = "education"
    else:
        return None

    unit = (source.get("unit") or "").strip() or "unknown"
    return category, int(year_match.group(0)), float(amount), unit


def _build_summary(
    verdict: str,
    category_a: str,
    category_b: str,
    year: int | None,
    fact_index: Dict[Tuple[str, int], Dict[str, Any]],
) -> str:
    if year is None:
        return f"FOL reasoning returned {verdict.lower()} for spending comparison between {category_a} and {category_b}."

    left = fact_index.get((category_a, year), {})
    right = fact_index.get((category_b, year), {})
    left_amount = left.get("amount")
    right_amount = right.get("amount")
    if left_amount is None or right_amount is None:
        return f"FOL reasoning returned {verdict.lower()} for {category_a} vs {category_b} in {year}."
    relation = "exceeded" if verdict == "Supported" else "did not exceed"
    return (
        f"FOL check found {category_a} spending ({left_amount:,.1f}) {relation} "
        f"{category_b} spending ({right_amount:,.1f}) in {year}."
    )


def _build_links(
    category_a: str,
    category_b: str,
    year: int | None,
    fact_index: Dict[Tuple[str, int], Dict[str, Any]],
) -> List[Dict[str, str]]:
    if year is None:
        return []
    links = []
    for category in (category_a, category_b):
        fact = fact_index.get((category, year))
        if not fact:
            continue
        if not fact.get("url"):
            continue
        links.append(
            {
                "finding": f"{fact.get('line_description', category)} ({year}) = {fact.get('amount', 'N/A')}",
                "source_url": fact["url"],
            }
        )
    return links

