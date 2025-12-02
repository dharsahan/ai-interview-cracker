import os
from openai import OpenAI
from typing import Optional

class LLMClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            print("No API Key provided. LLM will run in Mock Mode.")

    def get_answer(self, question: str) -> str:
        """
        Generates an answer for the given question.
        """
        if not question:
            return ""

        if not self.client:
            return self._mock_answer(question)

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", # Cost-effective for demo
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for job interviews. Keep answers concise and to the point. Structure them clearly."},
                    {"role": "user", "content": question}
                ],
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error contacting OpenAI: {e}"

    def _mock_answer(self, question: str) -> str:
        """Mock answer generator."""
        return f"[MOCK AI ANSWER] Here is a suggested answer for: '{question}'\n\n" \
               "1. Start with a clear definition.\n" \
               "2. Provide an example from your experience.\n" \
               "3. Conclude with the impact.\n\n" \
               "(Please configure OPENAI_API_KEY to get real AI responses)"
