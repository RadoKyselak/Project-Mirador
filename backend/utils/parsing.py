import json
import re
from typing import Any, Optional, Dict


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid JSON object from text."""
    if not text:
        return None
    
    start = text.find("{")
    if start == -1:
        return None
    
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    try:
                        cleaned = re.sub(r"[\x00-\x1f]", "", candidate)
                        return json.loads(cleaned)
                    except Exception:
                        return None
    return None

def parse_numeric_value(val: Any) -> Optional[float]:
    """Parse a numeric value from various string formats."""
    if val is None:
        return None
    try:
        s = str(val).strip().replace(",", "").replace("$", "")
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        m = re.match(r"^(-?[\d\.eE+-]+)", s)
        return float(m.group(1)) if m else float(s)
    except (ValueError, TypeError):
        return None


def apply_multiplier(value: Optional[float], multiplier: Optional[Any]) -> Optional[float]:
    """Apply a multiplier to a numeric value."""
    if value is None:
        return None
    if multiplier is None:
        return value
    try:
        return float(value) * float(multiplier)
    except (ValueError, TypeError):
        return value
