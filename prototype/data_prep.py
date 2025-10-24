import os
from dataclasses import dataclass
from typing import Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

SEASON_TO_MONTHS = {
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
    "fall": [9, 10, 11],
    "winter": [12, 1, 2],
}

PARK_ALIASES = {
    "Hong Kong": ["hong kong", "hk"],
    "California": ["california", "anaheim", "la"],
    "Paris": ["paris", "france"],
}


@dataclass
class DataIndex:
    df: pd.DataFrame
    vectorizer: TfidfVectorizer
    matrix: Any  # scipy sparse

    @staticmethod
    def _pick_col(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
        cols = list(df.columns)
        low = [c.lower() for c in cols]
        for kw in keywords:
            for i, lc in enumerate(low):
                if kw in lc:
                    return cols[i]
        return None

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        # Try multiple common names to be robust
        text_col = DataIndex._pick_col(df, ["review", "text", "body", "content"]) or df.columns[0]
        rating_col = DataIndex._pick_col(df, ["rating", "score", "stars"])
        country_col = DataIndex._pick_col(df, ["location", "country", "reviewer"])
        park_col = DataIndex._pick_col(df, ["branch", "park", "resort"])
        date_col = DataIndex._pick_col(df, ["year_month", "date", "year"])

        out = pd.DataFrame()
        out["text"] = df[text_col].astype(str)

        out["rating"] = df[rating_col] if rating_col else np.nan
        out["country"] = df[country_col].astype(str).str.strip() if country_col else ""
        out["park"] = df[park_col].astype(str).str.strip() if park_col else ""

        if date_col:
            ts = pd.to_datetime(df[date_col].astype(str), errors="coerce")
            out["year_month"] = ts
            out["month"] = ts.dt.month
            out["year"] = ts.dt.year
        else:
            out["month"] = np.nan
            out["year"] = np.nan

        def map_park(p: str) -> str:
            p_low = str(p).lower()
            for k, aliases in PARK_ALIASES.items():
                if any(a in p_low for a in aliases) or k.lower() in p_low:
                    return k
            return str(p).title()

        out["park_norm"] = out["park"].map(map_park)
        out["country_norm"] = out["country"].str.title()

        out = out.dropna(subset=["text"]).reset_index(drop=True)
        if out.empty:
            raise ValueError("Normalize produced empty DataFrame (no text).")
        return out

    @classmethod
    def from_csv(cls, path: str, use_sample: bool = False) -> "DataIndex":
        if use_sample or not os.path.exists(path):
            df_raw = pd.DataFrame({
                "Review_Text": [
                    "Amazing staff and clean park. Loved the parades!",
                    "Too crowded in June, lines were very long.",
                    "Visited in spring; weather was pleasant and flowers blooming.",
                    "As an Australian, Hong Kong Disney felt compact but charming.",
                ],
                "Rating": [5, 3, 4, 4],
                "Reviewer_Location": ["USA", "USA", "UK", "Australia"],
                "Branch": ["Paris", "California", "Paris", "Hong Kong"],
                "Year_Month": ["2019-12", "2018-06", "2019-04", "2017-11"],
            })
        else:
            df_raw = pd.read_csv(path)

        df = cls._normalize(df_raw)

        # min_df=1 so tiny samples work; adjust higher when using real CSV
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)
        matrix = vectorizer.fit_transform(df["text"].fillna(""))

        return cls(df=df, vectorizer=vectorizer, matrix=matrix)

    @classmethod
    def main(cls) -> None:
        path = os.getenv("DATA_PATH", "data/disneyland_reviews.csv")
        use_sample = bool(int(os.getenv("USE_SAMPLE_DATA", "1")))
        idx = cls.from_csv(path, use_sample=use_sample)
        print("Rows:", len(idx.df))
        print("Matrix:", idx.matrix.shape)
        print(idx.df.head(3)[["text", "rating", "country_norm", "park_norm", "month", "year"]])


if __name__ == "__main__":
    DataIndex.main()
