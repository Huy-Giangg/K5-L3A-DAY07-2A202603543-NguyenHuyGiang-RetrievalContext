import os
import json
import re
import dotenv
from openai import OpenAI

dotenv.load_dotenv()

def main():
    api_key = os.getenv("GROQ_API_KEY")
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    
    for model in ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]:
        print(f"\n--- Testing model: {model} ---")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a RAG controller. You must return ONLY valid JSON: {\"action\": \"search\", \"query\": \"...\"} or {\"action\": \"answer\", \"answer\": \"...\", \"citations\": [...]}."},
                    {"role": "user", "content": "Question: What was the debt-to-income ratio of IDA-eligible countries in 2025?\nHistory: No previous searches."}
                ],
                temperature=0.0
            )
            raw = resp.choices[0].message.content
            print("Response:", raw[:200])
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                print("Parsed successfully:", json.loads(m.group(0)))
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
