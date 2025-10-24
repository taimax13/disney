# Disney Reviews NLQ Prototype

    (🎢 Disney Reviews NLQ Prototype

A minimal production-oriented prototype demonstrating Natural Language Querying (NLQ) over Disneyland customer reviews.

📊 Overview

This prototype allows users to ask natural language questions such as:

“What do visitors from Australia say about Disneyland in Hong Kong?”

“Is spring a good time to visit Disneyland?”

“Is Disneyland California usually crowded in June?”

“Is the staff in Paris friendly?”

The system retrieves relevant reviews, applies metadata filters (country, park, date), and summarizes answers using either a heuristic offline summarizer or an optional LLM (4o-mini) if an API key is provided.

🧩 Dataset

Use the Kaggle dataset:
👉 Disneyland Reviews Dataset

After downloading, place it here:

data/disneyland_reviews.csv


If no dataset is found, the service automatically uses a sample dataset for quick testing.

⚙️ Quickstart
1. Environment setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

2. (Optional) Add OpenAI key

If you want to use the 4o-mini LLM instead of heuristic summaries:

export OPENAI_API_KEY="sk-xxxx..."


Otherwise, it runs fully offline.

3. Run the API
uvicorn prototype.app:app --reload

4. Health check
curl -s http://127.0.0.1:8000/healthz | jq


Example output:

{
  "status": "ok",
  "docs": 4
}

5. Query examples
curl -s --get 'http://127.0.0.1:8000/ask'   --data-urlencode 'q=Is the staff in Paris friendly?' | jq


or:

curl -s --get 'http://127.0.0.1:8000/ask'   --data-urlencode 'q=Is Disneyland California usually crowded in June?' | jq

6. Metrics
curl -s http://127.0.0.1:8000/metrics | head

🧠 Architecture 
```mermaid
flowchart LR

%% === Client Layer ===
subgraph Client
    U[Analyst UI / cURL] -->|HTTP JSON| API
end

%% === Service Layer ===
subgraph Service
    API[FastAPI NLQ API] --> MW[Logging & Metrics<br>Middleware]
    MW --> QP[Query Parser<br>(country / park / time filters)]
    QP --> RAG[Retriever<br>(TF-IDF + metadata)]
    RAG -->|top-k docs + metadata| LLM[LLMClient<br>(4o-mini or Heuristic)]
    RAG -->|telemetry| MON[Prometheus Exporter]
    MW --> MON
end

%% === Data Plane ===
subgraph DataPlane[Data Plane]
    DF[(Reviews CSV)] --> ETL[Loader + Preprocess]
    ETL --> IDX[Vector Index<br>(TF-IDF + metadata frame)]
    IDX --> RAG
end

%% === Monitoring / Dashboard ===
MON --> GRAF[Dashboard<br>(Prometheus / Grafana)]

```
🔍 Components
Component	Description
FastAPI	Exposes /ask, /healthz, and /metrics endpoints.
DataIndex	Loads, normalizes, and vectorizes review data.
Retriever	Uses TF-IDF + cosine similarity to find top relevant reviews.
LLMClient	Optional OpenAI 4o-mini call; otherwise uses a deterministic offline summarizer.
Monitoring	Prometheus metrics + JSON logging for latency, request counts, and errors.

🛡️ Monitoring Metrics

Prometheus endpoint /metrics exposes:

nlq_requests_total{endpoint}

nlq_request_latency_seconds{endpoint}

nlq_retrieved_docs_total{endpoint}

nlq_llm_failures_total

📈 Performance Notes

Average query latency: <100 ms retrieval + summarization.

Cold start: loads TF-IDF index once at startup.

Scales horizontally: index is read-only in memory.

🌐 Modes of Operation
Mode	Description
Offline / Heuristic (default)	No OPENAI_API_KEY. Summaries are based on ratings and top quotes.
Online / 4o-mini	Key is set → calls OpenAI API for richer answers. Fallbacks gracefully if API fails.

🧱 Folder structure
prototype/
 ├── app.py              # FastAPI entry point
 ├── data_prep.py        # Data loading + normalization + TF-IDF
 ├── rag.py              # Query parsing + retrieval
 ├── llm.py              # Optional LLM summarizer (offline fallback built-in)
 ├── monitoring.py       # Metrics + decorators
 ├── tests/
 │    └── test_app.py
 └── data/
      └── disneyland_reviews.csv

🧭 Example Workflow

User asks: “What do visitors from Australia say about Disneyland in Hong Kong?”

Query Parser extracts → country=Australia, park=Hong Kong.

Retriever → finds top-k relevant reviews (TF-IDF + metadata filter).

LLMClient → summarizes (LLM or heuristic).

Response → JSON with answer, evidence, and stats.

🧠 Future Improvements:

- Replace TF-IDF with embedding-based retrieval (e.g., E5 or text-embedding-3-small).

- Add quality evaluation pipeline and automated drift detection.

- Add frontend interface for analysts.)
    