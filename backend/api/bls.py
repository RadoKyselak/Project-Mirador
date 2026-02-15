import json
from typing import Dict, Any, List
import httpx
from config.constants import API_TIMEOUTS
from config import BLS_API_KEY, logger
from utils.parsing import parse_numeric_value


async def query_bls(params: Dict[str, Any]) -> List[Dict[str, Any]]:

    if not BLS_API_KEY:
        return [{"error": "BLS_API_KEY missing", "source": "BLS", "status": "failed"}]

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
            return [{"error": "BLS returned no data for series", "source": "BLS", "status": "no_data"}]

        annual_data = series_data[0].get("data", [])
        if not annual_data:
            logger.warning("BLS returned no annual data points for series %s, years %s-%s", series_id, start_year, end_year)
            return [{"error": "BLS returned no annual data points", "source": "BLS", "status": "no_data"}]

        year_values = {}
        for item in annual_data:
            if item.get("period") == "M13" and item.get("year") in [year_str, str(year_int - 1)]:
                value = parse_numeric_value(item.get("value"))
                if value is not None:
                    year_values[item.get("year")] = value

        if metric == "CPI":
            current_val = year_values.get(year_str)
            prev_val = year_values.get(str(year_int - 1))

            if current_val is None or prev_val is None:
                logger.warning(
                    f"BLS missing annual average CPI data for {year_str} or {year_int-1}. Available: {year_values}"
                )
                return [{
                    "error": f"BLS missing annual average CPI data for {year_str} or {year_int-1}",
                    "source": "BLS",
                    "status": "missing_data"
                }]
            if prev_val == 0:
                logger.error(f"BLS CPI data for previous year {year_int-1} is zero, cannot calculate change.")
                return [{
                    "error": f"BLS CPI data for {year_int-1} is zero, cannot calculate change.",
                    "source": "BLS",
                    "status": "calculation_error"
                }]

            percent_change = ((current_val - prev_val) / prev_val) * 100
            data_value = round(percent_change, 1)
            snippet = f"Annual average CPI inflation rate for {year_str} was {data_value}%."
            title = f"BLS: CPI Inflation Rate {year_str}"

        elif metric == "unemployment":
            data_value = year_values.get(year_str)
            if data_value is None:
                logger.warning(
                    f"BLS missing annual average unemployment data for {year_str}. Available: {year_values}"
                )
                return [{
                    "error": f"BLS missing annual average unemployment data for {year_str}",
                    "source": "BLS",
                    "status": "missing_data"
                }]

            data_value = round(data_value, 1)
            snippet = f"Annual average unemployment rate for {year_str} was {data_value}%."
            title = f"BLS: Unemployment Rate {year_str}"
        else:
            return [{"error": "Invalid BLS metric processing.", "source": "BLS", "status": "failed"}]

        return [{
            "title": title,
            "url": f"https://data.bls.gov/timeseries/{series_id}",
            "snippet": snippet,
            "data_value": data_value,
            "raw_data_value": str(data_value),
            "unit": "%"
        }]

    except Exception as e:
        logger.exception("Failed to parse BLS response")
        return [{"error": f"BLS parsing error: {str(e)}", "source": "BLS", "status": "failed"}]
