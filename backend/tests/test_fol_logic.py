from fol import Comparison, Constant, KnowledgeBase, Predicate, Rule, Variable, unify
from services.logic_verifier import fol_reason_about_claim


def test_unify_predicates_with_variable_binding():
    left = Predicate("Spending", [Variable("x"), Constant(2023)])
    right = Predicate("Spending", [Constant("defense"), Constant(2023)])
    result = unify(left, right)
    assert result is not None
    assert result["x"] == Constant("defense")


def test_kb_rule_derives_greater_spending():
    kb = KnowledgeBase()
    kb.add_fact(Predicate("Spending", [Constant("defense"), Constant(2023), Constant(790895.0), Constant("Millions")]))
    kb.add_fact(Predicate("Spending", [Constant("education"), Constant(2023), Constant(178621.0), Constant("Millions")]))

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

    assert kb.ask(Predicate("GreaterSpending", [Constant("defense"), Constant("education"), Constant(2023)]))
    assert not kb.ask(Predicate("GreaterSpending", [Constant("education"), Constant("defense"), Constant(2023)]))


def test_logic_verifier_supports_quantitative_comparison():
    claim = "Defense spending exceeded education spending in 2023."
    analysis = {
        "claim_type": "quantitative_comparison",
        "entities": ["defense", "education", "2023"],
        "relationship": "greater than",
    }
    sources = [
        {
            "url": "https://apps.bea.gov/api/data?x=1",
            "line_description": "National defense",
            "raw_year": "2023",
            "data_value": 790895.0,
            "unit": "Millions of Dollars",
        },
        {
            "url": "https://apps.bea.gov/api/data?x=2",
            "line_description": "Education",
            "raw_year": "2023",
            "data_value": 178621.0,
            "unit": "Millions of Dollars",
        },
    ]
    result = fol_reason_about_claim(claim, analysis, sources)
    assert result["verdict"] == "Supported"
    assert len(result["evidence_links"]) == 2


def test_logic_verifier_contradicts_inverted_comparison():
    claim = "Education spending exceeded defense spending in 2023."
    analysis = {
        "claim_type": "quantitative_comparison",
        "entities": ["education", "defense", "2023"],
        "relationship": "greater than",
    }
    sources = [
        {
            "url": "https://apps.bea.gov/api/data?x=1",
            "line_description": "National defense",
            "raw_year": "2023",
            "data_value": 790895.0,
            "unit": "Millions of Dollars",
        },
        {
            "url": "https://apps.bea.gov/api/data?x=2",
            "line_description": "Education",
            "raw_year": "2023",
            "data_value": 178621.0,
            "unit": "Millions of Dollars",
        },
    ]
    result = fol_reason_about_claim(claim, analysis, sources)
    assert result["verdict"] == "Contradicted"

