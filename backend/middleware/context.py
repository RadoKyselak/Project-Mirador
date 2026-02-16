import uuid
import time
from contextvars import ContextVar
from typing import Optional
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from config import logger

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
request_start_time_var: ContextVar[Optional[float]] = ContextVar("request_start_time", default=None)

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_id_var.set(request_id)
        
        start_time = time.time()
        request_start_time_var.set(start_time)
        
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None
            }
        )
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        logger.info(
            f"Request completed",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2)
            }
        )
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response

def get_request_id() -> Optional[str]:
    return request_id_var.get()

def get_request_duration() -> Optional[float]:
    start_time = request_start_time_var.get()
    if start_time:
        return time.time() - start_time
    return None
