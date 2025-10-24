import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List
from .data_prep import SEASON_TO_MONTHS, PARK_ALIASES
from .llm import LLMClient


COUNTRY_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b")
MONTHS = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
}


class NLQEngine:
    def __init__(self, data_index):
        self.idx = data_index
        self.llm = LLMClient()


    def _parse_filters(self, q: str) -> Dict:
        ql = q.lower()
        # park detection
        park = None
        for k, aliases in PARK_ALIASES.items():
            if k.lower() in ql or any(a in ql for a in aliases):
                park = k
                break
        country = None
        countries = set(self.idx.df["country_norm"].dropna().unique())
        for c in countries:
            if not isinstance(c, str):
                continue
            if c and c.lower() in ql:
                country = c
            break
        # month / season
        month = None
        for m, i in MONTHS.items():
            if m in ql:
                month = i
            break
        season_months = None
        for s, months in SEASON_TO_MONTHS.items():
            if s in ql:
                season_months = months
            break
        return {"park": park, "country": country, "month": month, "season_months": season_months}


    def _filter_df(self, filters: Dict):
        df = self.idx.df
        mask = np.ones(len(df), dtype=bool)
        if filters["park"]:
            mask &= (df["park_norm"] == filters["park"])
        if filters["country"]:
            mask &= (df["country_norm"] == filters["country"])
        if filters["month"]:
            mask &= (df["month"] == filters["month"])
        if filters["season_months"]:
            mask &= df["month"].isin(filters["season_months"]) if "month" in df else mask
        return df[mask]


    def _retrieve(self, q: str, cand_idx) -> List[Dict]:
        if len(cand_idx) == 0:
            return []
        q_vec = self.idx.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.idx.matrix[cand_idx])[0]
        order = np.argsort(-sims)
        topn = min(10, len(order))
        results = []
        for i in order[:topn]:
            row_id = cand_idx[i]
            row = self.idx.df.iloc[row_id]
        results.append({
        "text": row["text"],
        "rating": None if np.isnan(row.get("rating", np.nan)) else float(row.get("rating")),
        "park": row.get("park_norm"),
        "country": row.get("country_norm"),
        "month": int(row.get("month")) if not np.isnan(row.get("month", np.nan)) else None,
        })
        return results


    def answer(self, q: str) -> Dict:
        filters = self._parse_filters(q)
        df_f = self._filter_df(filters)
        cand_idx = df_f.index.to_list()
        snippets = self._retrieve(q, cand_idx)
        # record retrieval size for metrics externally (done in handler)
        answer = self.llm.summarize(q, snippets)
        stats = {
            "filters": filters,
            "candidates": len(cand_idx),
            "used": len(snippets)
        }
        return {"answer": answer, "evidence": snippets, "stats": stats}