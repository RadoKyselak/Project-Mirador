from typing import Dict, Any
from fastapi import HTTPException

from config import logger
from utils.parsing import extract_json_block
from .llm import call_gemini

async def analyze_claim_for_api_plan(claim: str) -> Dict[str, Any]:
    """
    Analyze a claim and generate an API query plan using LLM.
    Args:
        claim: The user's claim to analyze  
    Returns:
        Dictionary containing normalized claim, type, entities, and API plan
    """
    prompt_template = """
    You are a research analyst expert in U.S. government data APIs (BEA, Census, BLS, Data.gov, Congress.gov).
    Your goal is to analyze the user's claim and generate the *best possible single plan* to verify it using these APIs.

    **Analysis Steps:**
    1.  **Normalize Claim:** Rephrase the claim into a clear, verifiable statement (`claim_normalized`).
    2.  **Identify Entities:** List the specific concepts, agencies, metrics, locations, and timeframes mentioned (`entities`).
    3.  **Determine Claim Type:** Classify the claim (`claim_type`: quantitative_comparison, quantitative_value, factual, legislative, economic_indicator, other).
    4.  **Identify Relationship:** State the link asserted (e.g., "greater than", "value is X") (`relationship`).
    5.  **CRITICAL - Select API Strategy:** Based on the entities, choose the *most appropriate* API(s):
        * **Specific Agency/Department Budget/Spending?** (e.g., "Dept of Education budget", "NASA funding"): Primarily use `tier2_keywords` for `Data.gov`. **Do NOT use BEA.**
        * **Broad Economic Function Spending?** (e.g., "total spending on national defense function", "total education spending"): Use `BEA` (T31600).
        * **Demographic Data?** (e.g., "population of Alabama", "median income"): Use `Census ACS`.
        * **Inflation (CPI) or Unemployment Rate?**: Use `BLS`.
        * **Legislation/Bills?** (e.g., "CHIPS Act funding"): Primarily use `tier2_keywords` for `Congress.gov`.
        * **Other Specific Reports/Data/Topics?**: Use `tier2_keywords` for `Data.gov`.
    6.  **Generate API Plan:** Create the `api_plan` JSON. Fill `tier1_params` for BEA/Census/BLS if chosen. Fill `tier2_keywords` for Data.gov/Congress searches. If a specific API is chosen for Tier 1, still include relevant Tier 2 keywords as a backup or for context.

    **AVAILABLE APIs & Parameters:**
    -   `bea`: For broad *functions*. Needs `DataSetName` ("NIPA"), `TableName` ("T31600"), `Frequency` ("A"), `Year`, `LineCode` (e.g., "2" for Defense, "14" for Education).
    -   `census_acs`: For demographics. Needs `year`, `dataset` (e.g., "acs/acs1/profile"), `get` (vars, e.g., "NAME,DP05_0001E"), `for` (geo, e.g., "state:01").
    -   `bls`: For CPI/Unemployment. Needs `metric` ("CPI" or "unemployment") and `year`.
    -   `tier2_keywords`: List of search strings for Data.gov (budgets, reports) and Congress.gov (bills).

    **USER CLAIM:** '''{claim}'''

    **Return ONLY a single valid JSON object containing `claim_normalized`, `claim_type`, `entities`, `relationship`, and `api_plan`.**

    **Example (Agency Budget - Use Tier 2):**
    Claim: "The Department of Defense budget was over $800 billion in 2023."
    {{
      "claim_normalized": "The Department of Defense budget exceeded $800 billion in 2023.",
      "claim_type": "quantitative_value",
      "entities": ["Department of Defense Budget", "2023"],
      "relationship": "greater than",
      "api_plan": {{
        "tier1_params": {{ "bea": null, "census_acs": null, "bls": null }},
        "tier2_keywords": ["Department of Defense budget 2023", "FY2023 defense appropriations summary"]
      }}
    }}

    **Example (Broad Function - Use BEA):**
    Claim: "Total federal spending on the function of defense was more than education in 2023."
    {{
      "claim_normalized": "Total federal spending on the function of defense exceeded total federal spending on the function of education in 2023.",
      "claim_type": "quantitative_comparison",
      "entities": ["Federal Defense Function Spending", "Federal Education Function Spending", "2023"],
      "relationship": "greater than",
      "api_plan": {{
        "tier1_params": {{
          "bea": {{"DataSetName":"NIPA","TableName":"T31600","Frequency":"A","Year":"2023","LineCode":["2", "14"]}},
          "census_acs": null, "bls": null
        }},
        "tier2_keywords": ["federal spending by function 2023"]
      }}
    }}

    **Example (Inflation - Use BLS):**
    Claim: "Inflation was over 3% in 2023."
      {{
      "claim_normalized": "The US CPI inflation rate exceeded 3% in 2023.",
      "claim_type": "economic_indicator",
      "entities": ["US CPI Inflation", "2023"],
      "relationship": "greater than",
      "api_plan": {{
        "tier1_params": {{
          "bea": null, "census_acs": null,
          "bls": {{"metric": "CPI", "year": "2023"}}
        }},
        "tier2_keywords": ["US inflation rate 2023 annual average"]
      }}
    }}
    """
    prompt = prompt_template.format(claim=claim)
    fallback_plan = {
        "claim_normalized": claim,
        "claim_type": "Other",
        "entities": [],
        "relationship": "unknown",
        "api_plan": {"tier1_params": {}, "tier2_keywords": [claim, "general government data"]}
    }
    
    try:
        res = await call_gemini(prompt)
        parsed = extract_json_block(res.get("text", ""))

        if not parsed or "api_plan" not in parsed:
            logger.warning("Could not parse valid plan JSON from LLM. Falling back.")
            fallback_plan["debug_raw_llm_response"] = res.get("text", "No text found.")
            return fallback_plan

        parsed.setdefault("claim_normalized", claim)
        parsed.setdefault("claim_type", "Other")
        parsed.setdefault("entities", [])
        parsed.setdefault("relationship", "unknown")
        parsed.setdefault("api_plan", {"tier1_params": {}, "tier2_keywords": [claim]})
        parsed["api_plan"].setdefault("tier1_params", {})
        parsed["api_plan"]["tier1_params"].setdefault("bea", None)
        parsed["api_plan"]["tier1_params"].setdefault("census_acs", None)
        parsed["api_plan"]["tier1_params"].setdefault("bls", None)
        kw = parsed["api_plan"].get("tier2_keywords")
        parsed["api_plan"]["tier2_keywords"] = kw if isinstance(kw, list) and kw else [claim]

        return parsed
        
    except HTTPException as e:
        logger.error("LLM failed generating plan: %s", getattr(e, "detail", str(e)))
        fallback_plan["debug_exception"] = str(e)
        return fallback_plan
    except Exception as e:
        logger.exception("Unexpected error generating plan. Falling back.")
        fallback_plan["debug_exception"] = str(e)
        return fallback_plan
