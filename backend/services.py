import asyncio
import json
import httpx
from typing import List, Dict, Any
from config import settings
from schemas import ClaimAnalysis, APIPlan, Source
from data_sources import query_bea, query_census, query_congress, query_datagov
from prompts import ANALYSIS_PROMPT, SUMMARY_PROMPT

async def call_gemini(prompt: str) -> Dict[str, Any]:
    """Helper function to call the Gemini API."""
    if not settings.GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")
    
    headers = {"Content-Type": "application/json", "x-goog-api-key": settings.GEMINI_API_KEY}
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(settings.GEMINI_ENDPOINT, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return {"raw": data, "text": text}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e.response.status_code} - {e.response.text}")
    except (httpx.RequestError, json.JSONDecodeError, IndexError) as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Gemini: {str(e)}")

async def analyze_claim(claim: str) -> ClaimAnalysis:
    """Calls the LLM to get a structured API query plan."""
    prompt = ANALYSIS_PROMPT.format(claim=claim)
    res = await call_gemini(prompt)
    text = res["text"].strip().replace("```json", "").replace("```", "")
    
    try:
        data = json.loads(text)
        return ClaimAnalysis(
            claim_normalized=data.get("claim_normalized", claim),
            claim_type=data.get("claim_type", "unknown"),
            api_plan=APIPlan(
                tier1_params=data.get("api_plan", {}).get("tier1_params", {}),
                tier2_keywords=data.get("api_plan", {}).get("tier2_keywords", [claim])
            )
        )
    except Exception:
        return ClaimAnalysis(
            claim_normalized=claim,
            claim_type="qualitative",
            api_plan=APIPlan(tier2_keywords=[claim])
        )

def _pick_sources_from_type(claim_type: str) -> List[str]:
    """Helper to decide which APIs to query based on claim type."""
    mapping = {
        "quantitative": ["BEA", "CENSUS"], 
        "factual": ["CONGRESS"], 
        "default": ["DATA.GOV"]
    }
    return mapping.get(claim_type, mapping.get("default"))

async def gather_evidence(plan: APIPlan, claim_type: str) -> List[Source]:
    """Executes the API plan using the decoupled data source clients."""
    tier1_params = plan.tier1_params or {}
    tier2_keywords = plan.tier2_keywords or []
    sources_to_query = _pick_sources_from_type(claim_type)
    tasks, results = [], []
    
    for source_name in sources_to_query:
        if source_name == "BEA" and tier1_params.get("bea"):
            tasks.append(query_bea(params=tier1_params.get("bea")))
        if source_name == "CENSUS" and tier1_params.get("census"):
            tasks.append(query_census(params=tier1_params.get("census")))

    if tasks:
        tier1_results = await asyncio.gather(*tasks)
        results.extend([item for sublist in tier1_results for item in sublist if sublist])

    if not results and tier2_keywords:
        tasks = []
        for keyword in tier2_keywords:
            for source_name in sources_to_query:
                if source_name == "CENSUS":
                    tasks.append(query_census(keyword_query=keyword))
                if source_name == "CONGRESS":
                    tasks.append(query_congress(keyword_query=keyword))
        
        if tasks:
            tier2_results = await asyncio.gather(*tasks)
            results.extend([item for sublist in tier2_results for item in sublist if sublist])

    if not results and tier2_keywords:
        results.extend(await query_datagov(tier2_keywords[0]))
        
    return [Source(**s) for s in results]

async def synthesize_summary(claim: str, sources: List[Source]) -> str:
    """Calls the LLM with aggregated evidence to get a final summary."""
    if not sources:
        return "No supporting government data could be found to verify this claim."

    unique_sources = {s.url: s for s in sources}.values()
    context = "\n---\n".join([f"Source Title: {s.title}\nURL: {s.url}\nSnippet: {s.snippet}" for s in unique_sources])
    
    prompt = SUMMARY_PROMPT.format(claim=claim, context=context)
    res = await call_gemini(prompt)
    text = res["text"].strip().replace("```json", "").replace("```", "")
    
    try:
        parsed = json.loads(text)
        return f"{parsed.get('summary', 'Could not parse summary.')} {parsed.get('justification', '')}"
    except Exception:
        return "Could not generate a conclusive summary based on the available data."

def calculate_confidence(sources: List[Source], plan: APIPlan) -> float:
    """Calculates confidence score based on whether evidence was found."""
    return 0.95 if sources else 0.5
