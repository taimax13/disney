import time
import hashlib
from functools import wraps
from prometheus_client import Counter, Histogram, CollectorRegistry


def init_metrics():
    registry = CollectorRegistry()
    req_counter = Counter(
        "nlq_requests_total", "Total requests", ["endpoint"], registry=registry
    )
    latency_hist = Histogram(
        "nlq_request_latency_seconds", "Request latency", ["endpoint"], registry=registry
    )
    retrieved_counter = Counter(
        "nlq_retrieved_docs_total", "Retrieved docs", ["endpoint"], registry=registry
    )
    llm_failures = Counter(
        "nlq_llm_failures_total", "LLM failures", registry=registry
    )
    metrics = {
        "req_counter": req_counter,
        "latency_hist": latency_hist,
        "retrieved_counter": retrieved_counter,
        "llm_failures": llm_failures,
    }
    return registry, metrics


def instrument(metrics):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            endpoint = fn.__name__
            metrics["req_counter"].labels(endpoint=endpoint).inc()
            start = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.time() - start
                metrics["latency_hist"].labels(endpoint=endpoint).observe(elapsed)
        return wrapper
    return decorator


def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]
