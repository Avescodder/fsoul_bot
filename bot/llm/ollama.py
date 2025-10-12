import ollama
import os
from typing import List, Tuple
from bot.llm.base import BaseLLM, LLMResponse
import re


class OllamaLLM(BaseLLM):
    def __init__(self):
        self.client = ollama.AsyncClient(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        self.model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    
    async def generate_answer(
        self, 
        question: str, 
        context: List[Tuple[str, str]] = None
    ) -> LLMResponse:
        """Генерирует ответ с оценкой уверенности"""
        
        system_prompt = """Ты - эксперт по иммиграции в Португалию. Отвечай на вопросы пользователей четко и по делу.

ВАЖНО: В конце КАЖДОГО ответа добавь строку с оценкой своей уверенности:
CONFIDENCE: [число от 0.0 до 1.0]

Где:
- 1.0 = полностью уверен, есть точная информация
- 0.7-0.9 = достаточно уверен, общая информация
- 0.4-0.6 = частично уверен, нужны уточнения
- 0.0-0.3 = не уверен, нужна помощь эксперта

Отвечай на русском языке."""

        user_prompt = f"Вопрос: {question}\n\n"
        
        if context and len(context) > 0:
            user_prompt += "Похожие вопросы из базы знаний:\n\n"
            for i, (q, a) in enumerate(context[:3], 1):
                user_prompt += f"{i}. Q: {q}\n   A: {a}\n\n"
        
        user_prompt += "Дай ответ с оценкой уверенности в конце."
        
        response = await self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.7,
                "num_predict": 500
            }
        )
        
        answer_text = response['message']['content']
        
        confidence = self._extract_confidence(answer_text)
        
        clean_answer = re.sub(r'\n*CONFIDENCE:.*$', '', answer_text, flags=re.IGNORECASE).strip()
        
        return LLMResponse(
            answer=clean_answer,
            confidence=confidence,
            reasoning=f"Context items: {len(context) if context else 0}"
        )
    
    def _extract_confidence(self, text: str) -> float:
        """Извлекает значение confidence из текста"""
        match = re.search(r'CONFIDENCE:\s*(0?\.\d+|1\.0|0|1)', text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.5
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Генерирует эмбеддинг через Ollama"""
        response = await self.client.embeddings(
            model=self.embedding_model,
            prompt=text
        )
        return response['embedding']