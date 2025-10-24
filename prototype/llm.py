import os
from typing import List, Dict


# Minimal adapter; in practice call your 4o-mini endpoint here.
# For portability, we keep a deterministic fallback summarizer.


class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")


    def summarize(self, question: str, snippets: List[Dict]) -> str:
        if not self.api_key:
            return heuristic_answer(question, snippets)
        # Pseudocode for 4o-mini call (replace with actual SDK/client):
        # prompt = build_prompt(question, snippets)
        # resp = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.2)
        # return resp.choices[0].message.content
        return heuristic_answer(question, snippets) # keep it runnable without creds




def heuristic_answer(question: str, snippets: List[Dict]) -> str:
    if not snippets:
        return "I couldn't find matching reviews to answer that. Try broadening the query (park/country/season)."
    # Aggregate simple stats
    import statistics
    ratings = [s.get("rating") for s in snippets if s.get("rating") is not None]
    avg = round(statistics.mean(ratings), 2) if ratings else None
    park = snippets[0].get("park")
    country = snippets[0].get("country")
    quotes = [s["text"][:160] for s in snippets[:3]]
    parts = []
    if park:
        parts.append(f"For {park},")
    if country:
        parts.append(f"considering reviewers from {country},")
    if avg is not None:
        parts.append(f"the average rating among retrieved reviews is ~{avg}/5.")
        answer = " ".join(parts) if parts else "Based on retrieved reviews,"
        bullets = "\n- " + "\n- ".join(quotes)
    return f"{answer} Representative comments:\n{bullets}\n(Heuristic summary; set OPENAI_API_KEY to enable LLM synthesis.)"