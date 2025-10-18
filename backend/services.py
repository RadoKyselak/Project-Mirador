import asyncio
import json
import httpx
from typing import List
from config import settings
from schemas import ClaimAnalysis, APIPlan, Source
from data_sources import DATA_SOURCE_MAP
from prompts import ANALYSIS_PROMPT, SUMMARY_PROMPT

async def analyze_claim(claim: str) -> ClaimAnalysis:
    """Calls the LLM to get a structured API query plan."""
    prompt = ANALYSIS_PROMPT.format(claim=claim)
    return ClaimAnalysis(
        claim_normalized=claim,
        claim_type="qualitative",
        api_plan=APIPlan(tier2_keywords=[claim])
    )

async def gather_evidence(plan: APIPlan, claim_type: str) -> List[Source]:
    """Executes the API plan using the decoupled data source clients."""
    return []

async def synthesize_summary(claim: str, sources: List[Source]) -> str:
    """Calls the LLM with aggregated evidence to get a final summary."""
    context = "\n---\n".join([f"Source: {s.title}\nSnippet: {s.snippet}" for s in sources])
    prompt = SUMMARY_PROMPT.format(claim=claim, context=context)
    return "Could not generate summary."

def calculate_confidence(sources: List[Source], plan: APIPlan) -> float:
    """Calculates confidence score with clear, named constants."""
    return 0.0
