import re
from typing import Optional
import html

class ValidationError(Exception):
    pass

class InputValidator:
    
    SQL_INJECTION_PATTERN = re.compile(
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|SCRIPT)\b)",
        re.IGNORECASE
    )
    
    XSS_PATTERNS = [
        re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"<iframe", re.IGNORECASE),
    ]
    
    CONTROL_CHARS_PATTERN = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
    
    @staticmethod
    def sanitize_claim(claim: str) -> str:
        if not claim:
            raise ValidationError("Claim cannot be empty")
        
        claim = claim.strip()
        
        if len(claim) < 3:
            raise ValidationError("Claim must be at least 3 characters long")
        
        if len(claim) > 5000:
            raise ValidationError("Claim cannot exceed 5000 characters")
        
        if InputValidator.SQL_INJECTION_PATTERN.search(claim):
            raise ValidationError("Claim contains suspicious SQL-like patterns")
        
        for pattern in InputValidator.XSS_PATTERNS:
            if pattern.search(claim):
                raise ValidationError("Claim contains suspicious HTML/JavaScript patterns")
        
        claim = InputValidator.CONTROL_CHARS_PATTERN.sub('', claim)
        
        claim = html.escape(claim)
        
        claim = re.sub(r'\s+', ' ', claim)
        
        return claim
    
    @staticmethod
    def validate_claim(claim: str) -> tuple[bool, Optional[str]]:
        try:
            InputValidator.sanitize_claim(claim)
            return True, None
        except ValidationError as e:
            return False, str(e)
    
    @staticmethod
    def sanitize_api_parameter(param: str, max_length: int = 500) -> str:
        if not param:
            return ""
        
        param = str(param).strip()
        
        if len(param) > max_length:
            param = param[:max_length]
        
        param = InputValidator.CONTROL_CHARS_PATTERN.sub('', param)
        
        return param
