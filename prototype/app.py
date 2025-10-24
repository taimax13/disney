from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os

from .data_prep import DataIndex, SEASON_TO_MONTHS, PARK_ALIASES
from .monitoring import init_metrics, instrument
from .rag import NLQEngine

from starlette.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="Disney Reviews NLQ")
REGISTRY, METRICS = init_metrics()


# Global singletons (built at startup)
DATA_PATH = os.getenv("DATA_PATH", "../data/DisneylandReviews.csv")
USE_SAMPLE = bool(int(os.getenv("USE_SAMPLE_DATA", "0")))


try:
    data_index = DataIndex.from_csv(DATA_PATH, use_sample=USE_SAMPLE)
    engine = NLQEngine(data_index)
except Exception as e:
    raise RuntimeError(f"Failed to initialize index: {e}")


class AskResponse(BaseModel):
    answer: str
    evidence: list
    stats: dict


@app.get("/healthz")
def healthz():
    return {"status": "ok", "docs": len(data_index.df)}


@app.get("/ask", response_model=AskResponse)
@instrument(METRICS)
def ask(q: str = Query(..., min_length=3)):
    try:
        result = engine.answer(q)
        return JSONResponse(result)
    except Exception as e:
        METRICS["llm_failures"].inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
