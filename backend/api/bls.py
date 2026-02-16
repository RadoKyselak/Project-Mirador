import json
from typing import Dict, Any, List
import httpx
from config.constants import API_TIMEOUTS, RATE_LIMITS_PER_SECOND
from config import BLS_API_KEY, logger
from utils.parsing import parse_numeric_value
from utils.retry import async_retry
from utils.rate_limiter import get_rate_limiter

_bls_limiter = get_rate_limiter("BLS", RATE_LIMITS_PER_SECOND.BLS)

@async_retry(max_attempts=3, exceptions=(httpx.HTTPError, httpx.TimeoutException))
async def query_bls(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not BLS_API_KEY:
        return [{"error": "BLS_API_KEY missing", "source": "BLS", "status": "failed"}]

    await _bls_limiter.acquire()

    metric = params.get("metric")
    year_str = params.get("year")
    
    if not metric or not year_str:
        return [{"error": "BLS query missing metric or year", "source": "BLS", "status": "failed"}]

    try:
        year_int = int(year_str)
    except ValueError:
        return [{"error": "BLS query invalid year", "source": "BLS", "status": "failed"}]

    series_map = {
        "CPI": "CUSR0000SA0",
        "unemployment": "LNS14000000"
    }

    series_id = series_map.get(metric)
    if not series_id:
        return [{"error": f"BLS metric '{metric}' not supported", "source": "BLS", "status": "failed"}]

    start_year = str(year_int - 1) if metric == "CPI" else year_str
    end_year = year_str

    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = json.dumps({
        "seriesid": [series_id],
        "startyear": start_year,
        "endyear": end_year,
        "registrationKey": BLS_API_KEY,
        "annualaverage": True
    })
    headers = {'Content-Type': 'application/json'}

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUTS.BLS) as client:
            r = await client.post(url, headers=headers, content=payload)
            r.raise_for_status()
            data = r.json()

            if data.get("status") != "REQUEST_SUCCEEDED":
                message = data.get("message", ["Unknown BLS error."])[0]
                logger.error(f"BLS API error: {message}")
                return [{"error": f"BLS API error: {message}", "source": "BLS", "status": "failed"}]
    except httpx.HTTPStatusError as e:
        logger.error("BLS HTTP error %s: %s", e.response.status_code, e.response.text)
        return [{"error": f"BLS API error: {e.response.status_code}", "source": "BLS", "status": "failed"}]
    except httpx.RequestError as e:
        logger.error("BLS request error: %s", str(e))
        return [{"error": str(e), "source": "BLS", "status": "failed"}]
    except json.JSONDecodeError:
        logger.error("BLS returned non-JSON response: %s", r.text[:200])
        return [{"error": "BLS API returned invalid JSON", "source": "BLS", "status": "failed"}]

    try:
        series_data = data.get("Results", {}).get("series", [])
        if not series_data:
            logger.warning("BLS returned no data for series %s, years %s-%s", series_id, start_year, end_year)
            return [{"error": "BLS returned no data for series", "source": "BLS", "status": "failed"}]
        
        annual_data = series_data[0].get("data", [])
        results = []

        if metric == "CPI":
            current_year_vals = [
                d for d in annual_data
                if d.get("year") == year_str and d.get("period") == "M13"
            ]
            prev_year_vals = [
                d for d in annual_data
                if d.get("year") == start_year and d.get("period") == "M13"
            ]

            if current_year_vals and prev_year_vals:
                current_cpi = parse_numeric_value(current_year_vals[0].get("value"))
                prev_cpi = parse_numeric_value(prev_year_vals[0].get("value"))

                if current_cpi is not None and prev_cpi is not None and prev_cpi != 0:
                    inflation_rate = ((current_cpi - prev_cpi) / prev_cpi) * 100
                    snippet = (
                        f"CPI in {year_str}: {current_cpi:.1f}, "
                        f"CPI in {start_year}: {prev_cpi:.1f}. "
                        f"Inflation rate: {inflation_rate:.2f}%"
                    )
                    results.append({
                        "title": f"BLS CPI Data {year_str}",
                        "url": f"https://data.bls.gov/timeseries/{series_id}",
                        "snippet": snippet,
                        "data_value": inflation_rate,
                        "raw_data_value": f"{inflation_rate:.2f}",
                        "raw_cpi_current": current_cpi,
                        "raw_cpi_prev": prev_cpi,
                        "year": year_str
                    })
            else:
                logger.warning("BLS CPI data incomplete for %s", year_str)
                return [{"error": f"BLS CPI data incomplete for {year_str}", "source": "BLS", "status": "failed"}]

        elif metric == "unemployment":
            annual_avg = [
                d for d in annual_data
                if d.get("year") == year_str and d.get("period") == "M13"
            ]
            if annual_avg:
                unemp_rate = parse_numeric_value(annual_avg[0].get("value"))
                snippet = f"Unemployment rate in {year_str}: {unemp_rate}%"
                results.append({
                    "title": f"BLS Unemployment Rate {year_str}",
                    "url": f"https://data.bls.gov/timeseries/{series_id}",
                    "snippet": snippet,
                    "data_value": unemp_rate,
                    "raw_data_value": str(unemp_rate),
                    "year": year_str
                })
            else:
                logger.warning("BLS unemployment data not found for %s", year_str)
                return [{"error": f"BLS unemployment data not found for {year_str}", "source": "BLS", "status": "failed"}]

        return results

    except Exception as e:
        logger.exception("Error processing BLS data")
        return [{
            "error": f"Error processing BLS data: {str(e)}",
            "source": "BLS",
            "status": "failed"
        }]
