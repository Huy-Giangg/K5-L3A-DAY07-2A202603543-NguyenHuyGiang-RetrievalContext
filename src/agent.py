"""Unified Agent Controller supporting Groq (fast) and Gemini.

Implements the Controller protocol from starter/agentic.py:
- Decide whether to execute a search query or produce a grounded answer.
- Strictly cites evidence [chunk-id] in the text and in the citations list.
- Abstains honestly when the retrieved context lacks sufficient evidence.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional
import dotenv

dotenv.load_dotenv()

SYSTEM_PROMPT = """You are a rigorous financial research assistant analyzing the World Bank International Debt Report 2026.
Your task is to answer user questions based STRICTLY on the retrieved context chunks provided in History.

You have two actions:
1. "search": When you need information to answer the question, or if current context is empty or insufficient.
   Format: {"action": "search", "query": "<focused search query keywords>"}

2. "answer": When you either:
   a) Have found sufficient evidence in the retrieved context to answer the question accurately.
      Format:
      {
        "action": "answer",
        "answer": "<direct factual answer citing chunk IDs like [worldbank-idr-2026_p21_para_001]>",
        "citations": ["worldbank-idr-2026_p21_para_001"]
      }
   b) Or after searching, the retrieved context DOES NOT contain evidence to answer the question (or the question is out of scope).
      Format:
      {
        "action": "answer",
        "answer": "Không đủ bằng chứng trong tài liệu để trả lời.",
        "citations": []
      }

STRICT GROUNDING RULES:
- NEVER guess, assume, or hallucinate facts not in the context chunks.
- Every citation ID in "citations" MUST appear in the provided context AND must be referenced in the text like [chunk-id].
- If the question is outside the scope of the document (e.g. asking for Vietnam, Japan, or 2026 EU aid), after searching if no evidence is found, you MUST return empty citations [] and state that evidence is insufficient.
- You must output ONLY a valid JSON object matching one of the formats above.
"""


class GroqController:
    """Controller implementing the Controller protocol using Groq API (super fast)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        from openai import OpenAI
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY không được tìm thấy trong .env hoặc biến môi trường.")
        self.client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key, timeout=30.0)
        self.model_name = model_name or os.getenv("GROQ_MODEL") or "qwen/qwen3.8-27b"
        self.temperature = temperature

    def decide(self, question: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        context_blocks = []
        if history:
            for idx, h in enumerate(history, start=1):
                queries = ", ".join(h.get("queries", []))
                ctx = h.get("context", "")
                context_blocks.append(
                    f"--- Search History #{idx} ---\nQueries executed: {queries}\nRetrieved Context:\n{ctx}"
                )
        else:
            context_blocks.append("No searches performed yet. Context is currently empty.")

        user_content = (
            f"QUESTION: {question}\n\n"
            f"CURRENT CONTEXT AND SEARCH HISTORY:\n"
            f"{chr(10).join(context_blocks)}\n\n"
            f"REMINDER: When answering, you MUST write the citation directly in the answer text using square brackets like [chunk_id], and list the same chunk ID in 'citations'.\n"
            f"Decide your next action. Respond ONLY with valid JSON."
        )

        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=self.temperature,
                )
                raw_text = response.choices[0].message.content.strip()
                # Find JSON block using regex
                m = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if not m:
                    raise ValueError(f"No JSON object found in response: {raw_text}")
                data = json.loads(m.group(0))

                action = data.get("action")
                if action == "search":
                    query = data.get("query", question).strip()
                    return {"action": "search", "query": query or question}
                elif action == "answer":
                    answer = data.get("answer", "")
                    citations = data.get("citations", [])
                    if not isinstance(citations, list):
                        citations = []
                    # Ensure citations appear as [chunk_id] in the answer text
                    for cid in citations:
                        if f"[{cid}]" not in answer:
                            answer = f"{answer} [{cid}]"
                    return {"action": "answer", "answer": answer, "citations": citations}
                else:
                    return {"action": "search", "query": question}
            except Exception as e:
                err_str = str(e).lower()
                is_transient = any(kw in err_str for kw in ["429", "rate", "connection", "timed out", "timeout", "503"])
                if is_transient and attempt < max_attempts - 1:
                    wait_s = 4 + attempt * 4
                    print(f"    [Transient API notice: {e}. Pausing {wait_s}s before retry {attempt+2}/{max_attempts}...]")
                    time.sleep(wait_s)
                    continue
                raise RuntimeError(f"Groq API error during controller decision: {e}")
        raise RuntimeError("Exceeded maximum retries for Groq API.")


class GeminiController:
    """Controller implementing the Controller protocol using Google Gemini API."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        from google import genai
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY không được tìm thấy trong .env hoặc biến môi trường.")
        self.client = genai.Client(api_key=key)
        self.model_name = model_name or os.getenv("GEMINI_MODEL") or "gemini-3.5-flash"
        self.temperature = temperature

    def decide(self, question: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        from google.genai import types
        context_blocks = []
        if history:
            for idx, h in enumerate(history, start=1):
                queries = ", ".join(h.get("queries", []))
                ctx = h.get("context", "")
                context_blocks.append(
                    f"--- Search History #{idx} ---\nQueries executed: {queries}\nRetrieved Context:\n{ctx}"
                )
        else:
            context_blocks.append("No searches performed yet. Context is currently empty.")

        user_content = (
            f"QUESTION: {question}\n\n"
            f"CURRENT CONTEXT AND SEARCH HISTORY:\n"
            f"{chr(10).join(context_blocks)}\n\n"
            f"Decide your next action. Respond ONLY with valid JSON."
        )

        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        temperature=self.temperature,
                    ),
                )
                raw_text = response.text.strip()
                if raw_text.startswith("```"):
                    raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.DOTALL).strip()
                data = json.loads(raw_text)

                action = data.get("action")
                if action == "search":
                    query = data.get("query", question).strip()
                    return {"action": "search", "query": query or question}
                elif action == "answer":
                    answer = data.get("answer", "")
                    citations = data.get("citations", [])
                    if not isinstance(citations, list):
                        citations = []
                    return {"action": "answer", "answer": answer, "citations": citations}
                else:
                    return {"action": "search", "query": question}
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                    wait_s = 15 + attempt * 10
                    print(f"    [Rate limit 429 hit. Pausing {wait_s}s before retry...]")
                    time.sleep(wait_s)
                    continue
                raise RuntimeError(f"Gemini API error during controller decision: {e}")
        raise RuntimeError("Exceeded maximum retries for Gemini API due to rate limits.")


def get_default_controller():
    """Factory creating controller based on PROVIDER in .env (defaults to Groq)."""
    provider = os.getenv("PROVIDER", "groq").lower()
    if provider == "groq" and os.getenv("GROQ_API_KEY"):
        return GroqController()
    elif provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        return GeminiController()
    # Fallback to available key
    if os.getenv("GROQ_API_KEY"):
        return GroqController()
    elif os.getenv("GEMINI_API_KEY"):
        return GeminiController()
    raise ValueError("Không tìm thấy GROQ_API_KEY hoặc GEMINI_API_KEY trong .env")
