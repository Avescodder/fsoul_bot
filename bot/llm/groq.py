import os
import re
from typing import List, Tuple
from bot.llm.base import BaseLLM, LLMResponse
from groq import Groq


class GroqLLM(BaseLLM):
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    async def generate_answer(
        self, 
        question: str, 
        context: List[Tuple[str, str]] = None
    ) -> LLMResponse:
        """Генерирует ответ через Groq API"""
        
        system_prompt = """Ты — Сергей, профессиональный консультант по иммиграции в Португалию.

СТИЛЬ ОБЩЕНИЯ:
- Общайся естественно, как реальный человек-специалист
- Будь дружелюбным, но профессиональным
- Используй структурированные ответы с разбивкой на пункты где это уместно
- Избегай слишком формального языка
- Не упоминай, что ты AI или бот

СТРУКТУРА ОТВЕТА:
1. Краткий прямой ответ на вопрос
2. Подробное объяснение с конкретными деталями
3. Практические рекомендации или следующие шаги (если применимо)
4. Дополнительная полезная информация (если есть)

Используй форматирование:
- **Жирный текст** для важных моментов
- Нумерованные списки для последовательных шагов
- Маркированные списки для перечислений
- Абзацы для лучшей читаемости

ОЦЕНКА УВЕРЕННОСТИ:
В конце КАЖДОГО ответа ОБЯЗАТЕЛЬНО добавь строку:
CONFIDENCE: [число от 0.0 до 1.0]

Критерии уверенности:
- **0.9-1.0**: Точная информация из базы знаний, полностью уверен
- **0.7-0.8**: Хорошее понимание темы, общая информация корректна
- **0.5-0.6**: Частичная уверенность, информация может требовать уточнений
- **0.3-0.4**: Низкая уверенность, нужна консультация с коллегами
- **0.0-0.2**: Не уверен, требуется помощь эксперта

ВАЖНО: 
- Всегда отвечай на русском языке
- Базируйся на информации из базы знаний
- Если информации недостаточно, честно признай это и поставь низкую уверенность
- Не придумывай факты, лучше признать незнание"""

        user_prompt = f"Вопрос клиента: {question}\n\n"
        
        if context and len(context) > 0:
            user_prompt += "📚 Релевантная информация из базы знаний:\n\n"
            for i, (q, a) in enumerate(context[:3], 1):
                user_prompt += f"Пример {i}:\n"
                user_prompt += f"Вопрос: {q}\n"
                user_prompt += f"Ответ: {a}\n\n"
            user_prompt += "---\n\n"
        else:
            user_prompt += "⚠️ В базе знаний не найдено похожих вопросов. Отвечай на основе общих знаний, но будь осторожен с уверенностью.\n\n"
        
        user_prompt += "Дай структурированный ответ от имени Сергея с оценкой уверенности в конце."
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=1500,
                top_p=1,
                stream=False
            )
            
            content = chat_completion.choices[0].message.content
            
            confidence = self._extract_confidence(content)
            
            clean_answer = re.sub(
                r'\n*CONFIDENCE:\s*[\d\.]+\s*', 
                '', 
                content, 
                flags=re.IGNORECASE
            ).strip()
            
            return LLMResponse(
                answer=clean_answer,
                confidence=confidence,
                reasoning=f"Groq API ({self.model}), context items: {len(context) if context else 0}"
            )
            
        except Exception as e:
            print(f"❌ Groq API error: {e}")
            return LLMResponse(
                answer="Извините, произошла техническая ошибка. Пожалуйста, повторите вопрос через несколько минут.",
                confidence=0.0,
                reasoning=f"Error: {str(e)}"
            )
    
    def _extract_confidence(self, text: str) -> float:
        """Извлекает значение confidence из текста"""
        match = re.search(r'CONFIDENCE:\s*(0?\.\d+|1\.0|0|1)', text, re.IGNORECASE)
        if match:
            try:
                confidence_value = float(match.group(1))
                return max(0.0, min(1.0, confidence_value))
            except ValueError:
                pass
        
        print("⚠️ CONFIDENCE не найдена в ответе LLM, используем 0.5")
        return 0.5
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Groq не предоставляет embeddings API.
        Используем локальную модель sentence-transformers.
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            if not hasattr(self, '_embedding_model'):
                print("🔄 Загрузка embedding модели...")
                self._embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            
            embedding = self._embedding_model.encode(text)
            return embedding.tolist()
            
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install it with: pip install sentence-transformers"
            )
        except Exception as e:
            print(f"❌ Embedding generation error: {e}")
            return [0.0] * 768