
# Disney Reviews NLQ Prototype


## Dataset
Use the Kaggle dataset: https://www.kaggle.com/datasets/arushchillar/disneyland-reviews
Place the CSV as `data/disneyland_reviews.csv`.


## Run
Health:
-  curl -s http://127.0.0.1:8000/healthz | jq
```{
  "status": "ok",
  "docs": 4
}
```

see Quickstart in main doc. Example queries:
- "What do visitors from Australia say about Disneyland in Hong Kong?"
- "Is spring a good time to visit Disneyland?"
- "Is Disneyland California usually crowded in June?"
- "Is the staff in Paris friendly?"

f.e:
```curl -s --get 'http://127.0.0.1:8000/ask' --data-urlencode 'q=Is the staff in Paris friendly?' | jqCC
```


```mermaid
flowchart LR
subgraph Client
U[Analyst UI / cURL] -->|HTTP /json| API
end


subgraph Service
API[FastAPI NLQ API] --> MW[Logging & Metrics Middleware]
MW --> QP[Query Parser\n(country/park/time filters)]
QP --> RAG[Retriever]
RAG -->|top‑k docs + metadata| LLM[4o‑mini LLM\n(answer synthesis)]
RAG -->|telemetry| MON[Prometheus Exporter]
MW --> MON
end


subgraph Data Plane
DF[(Reviews CSV)] --> ETL[Loader + Preprocess]
ETL --> IDX[Vector Index (TF‑IDF)\n+ metadata frame]
IDX --> RAG
end


MON --> GRAF[Dash/Prom/Grafana]

```

