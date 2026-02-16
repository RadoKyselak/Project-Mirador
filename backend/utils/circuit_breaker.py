import asyncio
import time
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps
from config import logger

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        name: Optional[str] = None
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or "unnamed"
        
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = CircuitState.CLOSED
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state
    
    @property
    def failure_count(self) -> int:
        return self._failure_count

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info(
                        f"Circuit breaker {self.name}: Transitioning to half-open state",
                        extra={"circuit_breaker": self.name, "state": "half_open"}
                    )
                    self._state = CircuitState.HALF_OPEN
                else:
                    from exceptions import CircuitBreakerOpenException
                    logger.warning(
                        f"Circuit breaker {self.name} is open, rejecting request",
                        extra={
                            "circuit_breaker": self.name,
                            "failure_count": self._failure_count,
                            "state": "open"
                        }
                    )
                    raise CircuitBreakerOpenException(self.name, self._failure_count)

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self.expected_exception as e:
            await self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        return (
            self._last_failure_time is not None
            and time.time() - self._last_failure_time >= self.recovery_timeout
        )

    async def _on_success(self):
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info(
                    f"Circuit breaker {self.name}: Recovery successful, closing circuit",
                    extra={"circuit_breaker": self.name, "state": "closed"}
                )
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    async def _on_failure(self):
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker {self.name}: Failure threshold reached, opening circuit",
                    extra={
                        "circuit_breaker": self.name,
                        "failure_count": self._failure_count,
                        "threshold": self.failure_threshold,
                        "state": "open"
                    }
                )
                self._state = CircuitState.OPEN

def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception,
    name: Optional[str] = None
):
    breaker = CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception,
        name=name
    )
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        wrapper._circuit_breaker = breaker
        return wrapper
    return decorator
