import os
from typing import List, Dict

try:
    import openai  # only required if you actually want to call the model
except ImportError:
    openai = None


def heuristic_answer(question: str, snippets: List[Dict]) -> str:
    """
    Lightweight summarizer for offline mode (no API key).
    Aggregates basic sentiment from snippets and gives sample quotes.
    """
    if not snippets:
        return (
            "I couldn't find matching reviews to answer that. "
            "Try broadening the query (park/country/season)."
        )

    import statistics
    ratings = [s.get("rating") for s in snippets if s.get("rating") is not None]
    avg = round(statistics.mean(ratings), 2) if ratings else None
    park = snippets[0].get("park")
    country = snippets[0].get("country")
    quotes = [s["text"][:120] for s in snippets[:3]]

    answer_parts = []
    if park:
        answer_parts.append(f"For {park},")
    if country:
        answer_parts.append(f"based on visitors from {country},")
    if avg is not None:
        answer_parts.append(f"average rating is around {avg}/5.")
    answer = " ".join(answer_parts) or "Based on retrieved reviews,"

    bullets = "\n- " + "\n- ".join(quotes)
    return f"{answer} Representative comments:{bullets}\n(Heuristic summary; no LLM used.)"


class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if openai and self.api_key:
            openai.api_key = self.api_key
        self.active = bool(openai and self.api_key)

    def summarize(self, question: str, snippets: List[Dict]) -> str:
        if not self.active:
            # offline fallback
            return heuristic_answer(question, snippets)

        # --- Online mode using 4o-mini ---
        try:
            prompt = self.build_prompt(question, snippets)
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content
        except Exception as e:
            # fallback on any API failure
            print(f"[LLMClient] Fallback due to error: {e}")
            return heuristic_answer(question, snippets)

    @staticmethod
    def build_prompt(question: str, snippets: List[Dict]) -> str:
        joined = "\n".join(f"- {s['text']}" for s in snippets[:5])
        return (
            f"Answer the question below based only on these Disney park reviews.\n"
            f"Question: {question}\n\nReviews:\n{joined}\n\n"
            "Summarize concisely with key insights."
        )
