import os
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simplified environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Stelthar-API is running."}

class VerifyRequest(BaseModel):
    claim: str

async def call_gemini(prompt: str) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(GEMINI_ENDPOINT, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return {"raw": data, "text": text}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e.response.status_code} - {e.response.text}")
    except (httpx.RequestError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Gemini: {str(e)}")

async def analyze_claim_for_api_plan(claim: str) -> Dict[str, Any]:
    # Simplified prompt focusing only on keyword generation for Data.gov
    prompt = (
        "You are a world-class research analyst and a U.S. government data expert. Your task is to deconstruct a user's factual claim into a list of precise keyword queries to search on the Data.gov API.\n\n"
        f"USER CLAIM: '''{claim}'''\n\n"
        "YOUR RESPONSE (Must be a single, valid JSON object):\n"
        "{\n"
        '  "claim_normalized": "Your clear, verifiable statement of the claim.",\n'
        '  "keywords": ["A list of 3-5 specific keyword search queries to find relevant datasets on Data.gov."]\n'
        "}"
    )
    res = await call_gemini(prompt)
    text = res["text"].strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback if the model fails to generate valid JSON
        return {"claim_normalized": claim, "keywords": [claim]}

async def query_datagov(keyword_query: str) -> List[Dict[str, str]]:
    if not DATA_GOV_API_KEY:
        raise HTTPException(status_code=500, detail="DATA_GOV_API_KEY is not configured on the server.")
    if not keyword_query:
        return []
    params = {"api_key": DATA_GOV_API_KEY, "q": keyword_query, "limit": 5}
    url = "https://api.data.gov/catalog/v1"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return [{"title": item.get("title"), "url": item.get("@id"), "snippet": item.get("description", "")[:250]} for item in r.json().get("results", [])]
    except Exception as e:
        print(f"Data.gov API Error: {e}")
        return []

async def execute_query_plan(plan: Dict) -> List[Dict[str, Any]]:
    keywords = plan.get('keywords', [])
    if not keywords:
        return []
        
    tasks = [query_datagov(keyword) for keyword in keywords]
    query_results = await asyncio.gather(*tasks)
    
    # Flatten the list of lists and remove any empty results
    results = [item for sublist in query_results for item in sublist if sublist]
    return results
    
async def summarize_with_evidence(claim: str, sources: List[Dict[str, str]]) -> str:
    if not sources:
        return "No supporting government data could be found to verify this claim."
    # Deduplicate sources based on URL
    unique_sources = {s['url']: s for s in sources}.values()
    context = "\n---\n".join([f"Source Title: {s['title']}\nURL: {s['url']}\nSnippet: {s['snippet']}" for s in unique_sources])
    prompt = (
        "You are a meticulous and impartial fact-checker. Your sole responsibility is to analyze the provided evidence from U.S. government data sources and synthesize a definitive conclusion about the user's claim. Do not introduce outside information.\n\n"
        "YOUR METHODOLOGY (CHAIN-OF-THOUGHT):\n"
        "1. First, review all evidence snippets. Identify the key data points relevant to the claim.\n"
        "2. Second, compare the data points. Are they consistent? Do they contradict each other? Is there enough information to make a judgment?\n"
        "3. Third, synthesize your findings into a concise, one-sentence summary that directly addresses the claim. State whether the evidence supports, contradicts, or is insufficient to verify the claim. Start your summary with a clear concluding phrase (e.g., 'The data supports...', 'The data contradicts...', 'The available data is insufficient to...').\n"
        "4. Fourth, provide a brief (1-2 sentence) justification for your conclusion, citing the key pieces of evidence from the snippets.\n\n"
        f"USER'S CLAIM: '''{claim}'''\n\n"
        f"AGGREGATED EVIDENCE:\n{context}\n\n"
        "YOUR RESPONSE (Must be a single, valid JSON object):\n"
        "{\n"
        '  "summary": "Your final, synthesized one-sentence conclusion.",\n'
        '  "justification": "Your brief justification citing the evidence."\n'
        "}"
    )
    res = await call_gemini(prompt)
    text = res["text"].strip().replace("```json", "").replace("```", "")
    try:
        parsed = json.loads(text)
        return f"{parsed.get('summary', '')} {parsed.get('justification', '')}"
    except json.JSONDecodeError:
        return "Could not generate a conclusive summary based on the available data."

@app.post("/verify")
async def verify(req: VerifyRequest):
    claim = req.claim.strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Empty claim.")
    
    analysis = await analyze_claim_for_api_plan(claim)
    
    claim_norm = analysis.get("claim_normalized")
    
    sources_results = await execute_query_plan(analysis)
    
    verdict, confidence = ("Inconclusive", 0.5) if not sources_results else ("Verifiable", 0.95)
    summary = await summarize_with_evidence(claim_norm, sources_results)
    
    return {
        "claim_original": claim,
        "claim_normalized": claim_norm,
        "verdict": verdict,
        "confidence": confidence,
        "summary": summary,
        "sources": sources_results,
        "debug_plan": analysis
    }
