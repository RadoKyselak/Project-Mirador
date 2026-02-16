from typing import Optional, Dict, Any

class MiradorException(Exception):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }

class APIException(MiradorException):
    pass

class RateLimitException(APIException):
    def __init__(self, api_name: str, retry_after: float):
        super().__init__(
            f"Rate limit exceeded for {api_name}",
            {"api_name": api_name, "retry_after": retry_after}
        )

class ValidationException(MiradorException):
    def __init__(self, field: str, reason: str):
        super().__init__(
            f"Validation failed for {field}: {reason}",
            {"field": field, "reason": reason}
        )

class LLMException(APIException):
    def __init__(self, reason: str, recoverable: bool = True):
        super().__init__(
            f"LLM service error: {reason}",
            {"reason": reason, "recoverable": recoverable}
        )

class DataSourceException(APIException):
    def __init__(self, source: str, reason: str, recoverable: bool = True):
        super().__init__(
            f"Data source {source} failed: {reason}",
            {"source": source, "reason": reason, "recoverable": recoverable}
        )

class CircuitBreakerOpenException(APIException):
    def __init__(self, service_name: str, failure_count: int):
        super().__init__(
            f"Circuit breaker open for {service_name}",
            {"service": service_name, "failure_count": failure_count}
        )
