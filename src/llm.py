import os
import ollama
from typing import Optional

class LLMClient:
    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = None
        self._connected = False
        
        try:
            self.client = ollama.Client(host=self.host)
            # Test connection by listing models
            self.client.list()
            self._connected = True
        except Exception as e:
            print(f"Could not connect to Ollama at {self.host}. LLM will run in Mock Mode. Error: {e}")
            self._connected = False

    def get_answer(self, question: str) -> str:
        """
        Generates an answer for the given question.
        """
        if not question:
            return ""

        if not self._connected:
            return self._mock_answer(question)

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for job interviews. Keep answers concise and to the point. Structure them clearly."},
                    {"role": "user", "content": question}
                ]
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"Error contacting Ollama: {e}"

    def _mock_answer(self, question: str) -> str:
        """Mock answer generator."""
        return f"[MOCK AI ANSWER] Here is a suggested answer for: '{question}'\n\n" \
               "1. Start with a clear definition.\n" \
               "2. Provide an example from your experience.\n" \
               "3. Conclude with the impact.\n\n" \
               "(Please ensure Ollama is running with 'ollama serve' to get real AI responses)"
