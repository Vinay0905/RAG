import time 
from functools import wraps 
from typing import Callable,Any
from src.core.logging import logger

class TelemetryTracer:
    """Lightweight tracer for recording step latency and telemetry."""

    @staticmethod
    def trace_span(span_name: str) -> Callable:
        """Decorator to trace function execution duration and log metrics."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter()
                logger.info(f"⚡ [SPAN START] {span_name}")
                try:
                    result = func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(f"✅ [SPAN END] {span_name} completed in {elapsed_ms:.2f}ms")
                    return result
                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    logger.error(f"❌ [SPAN FAILED] {span_name} failed after {elapsed_ms:.2f}ms: {str(e)}")
                    raise e
            return wrapper
        return decorator


tracer = TelemetryTracer()