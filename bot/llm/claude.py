import anthropic
import os
from typing import List, Tuple
from bot.llm.base import BaseLLM, LLMResponse
import json


class ClaudeLLM(BaseLLM):
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        self.model = "claude-sonnet-4-20250514"
    
    async def generate_answer(
        self, 
        question: str, 
        context: List[Tuple[str, str]] = None
    ) -> LLMResponse:
        """Генерирует ответ через Claude API"""
        
        system_prompt = """Ты - эксперт по иммиграции в Португалию. Отвечай на вопросы пользователей четко и по делу на русском языке.

После ответа оцени свою уверенность по шкале от 0.0 до 1.0:
- 1.0 = полностью уверен, точная информация
- 0.7-0.9 = достаточно уверен
- 0.4-0.6 = частично уверен
- 0.0-0.3 = не уверен

Верни ответ в JSON формате:
{
  "answer": "твой ответ",
  "confidence": 0.85
}"""

        user_prompt = f"Вопрос: {question}\n\n"
        
        if context and len(context) > 0:
            user_prompt += "Похожие вопросы из базы знаний:\n\n"
            for i, (q, a) in enumerate(context[:3], 1):
                user_prompt += f"{i}. Q: {q}\n   A: {a}\n\n"
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.7,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        content = response.content[0].text
        
        try:
            data = json.loads(content)
            return LLMResponse(
                answer=data.get("answer", content),
                confidence=float(data.get("confidence", 0.5)),
                reasoning="Claude API response"
            )
        except json.JSONDecodeError:
            return LLMResponse(
                answer=content,
                confidence=0.6,
                reasoning="Fallback parsing"
            )
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Claude API не предоставляет embeddings напрямую.
        Используем Ollama для эмбеддингов или другой сервис.
        """
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        embedding = model.encode(text)
        return embedding.tolist()