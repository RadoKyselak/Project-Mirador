from typing import Dict, Any, List, Tuple
from config import logger
from utils.similarity import cosine_similarity
from .llm import get_embeddings_batch_api

async def compute_confidence(sources: List[Dict[str, Any]], verdict: str, claim: str) -> Dict[str, Any]:
    """
    Compute confidence score for the verification result.
    Args:
        sources: List of source data from APIs
        verdict: The verdict reached (Supported, Contradicted, Inconclusive)
        claim: Original user claim
    Returns:
        Dictionary containing confidence score and breakdown
    """
    valid_sources = [s for s in sources if s and "error" not in s]

    DEFAULT_CONFIDENCE_DATA = {"confidence": 0.3, "R": 0.5, "E": 0.0, "S": 0.3, "S_semantic_sim": 0.0}

    if not valid_sources:
        return DEFAULT_CONFIDENCE_DATA

    total_weight = 0.0
    for s in valid_sources:
        url = (s.get("url") or "").lower()
        if "apps.bea.gov" in url:
            weight = 1.0
        elif "api.census.gov" in url:
            weight = 1.0
        elif "api.bls.gov" in url:
            weight = 1.0
        elif "api.congress.gov" in url:
            weight = 0.8
        elif "catalog.data.gov" in url:
            weight = 0.7
        else:
            weight = 0.6
        total_weight += weight

    R = round(total_weight / len(valid_sources), 2) if valid_sources else 0.5
    
    E = round(min(1.0, len(valid_sources) / 5.0), 2)

    S_llm_verdict = 0.5
    if verdict == "Supported":
        S_llm_verdict = 0.95
    elif verdict == "Contradicted":
        S_llm_verdict = 0.90
    S_semantic_sim = 0.0

    try:
        texts_to_embed = [claim] if claim and claim.strip() else []
        if not texts_to_embed:
            raise ValueError("Claim text is empty, cannot compute semantic similarity.")

        for s in valid_sources:
            snippet = s.get('snippet', '').strip()
            title = s.get('title', '').strip()
            data_text = None
            if s.get('data_value') is not None:
                data_text = f"{s.get('line_description', s.get('title', 'Data'))}: {s.get('raw_data_value')}"
                data_text = data_text.strip()

            if snippet:
                texts_to_embed.append(snippet)
            if title:
                texts_to_embed.append(title)
            if data_text:
                texts_to_embed.append(data_text)

        if len(texts_to_embed) > 1:
            all_embeddings = await get_embeddings_batch_api(texts_to_embed)

            if not all_embeddings or all_embeddings[0] is None:
                logger.warning("Claim embedding failed, cannot calculate semantic similarity.")
                S_semantic_sim = 0.3
            else:
                claim_embedding = all_embeddings[0]
                source_embeddings = all_embeddings[1:]
                valid_source_embeddings = [emb for emb in source_embeddings if emb]

                if valid_source_embeddings:
                    similarities = [cosine_similarity(claim_embedding, emb) for emb in valid_source_embeddings]
                    if similarities:
                        S_semantic_sim = round(float(max(similarities)), 2)
                    else:
                        S_semantic_sim = 0.3
                else:
                    logger.warning("No valid source embeddings returned from batch API call.")
                    S_semantic_sim = 0.3

    except ValueError as ve:
        logger.error(f"Cannot compute semantic similarity: {ve}")
        S_semantic_sim = 0.1
    except Exception as e:
        logger.error("Error during batch semantic similarity calculation: %s", e, exc_info=True)
        S_semantic_sim = 0.3

    S = round((S_llm_verdict * 0.7) + (S_semantic_sim * 0.3), 2)
    confidence = round((0.5 * R + 0.3 * S + 0.2 * E), 2)

    return {"confidence": confidence, "R": R, "E": E, "S": S, "S_semantic_sim": S_semantic_sim}
