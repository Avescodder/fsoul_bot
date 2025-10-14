import os
import re
from typing import List, Tuple
from bot.llm.base import BaseLLM, LLMResponse
from openai import AsyncOpenAI


class OpenAILLM(BaseLLM):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    async def generate_answer(
        self, 
        question: str, 
        context: List[Tuple[str, str]] = None
    ) -> LLMResponse:
        """Генерирует ответ через OpenAI API"""
        
        system_prompt = """Ты — Сергей, профессиональный консультант по иммиграции в Португалию.

СТИЛЬ ОБЩЕНИЯ:
- Пиши как живой человек в мессенджере
- Без markdown форматирования (без **, без #, без списков с - или *)
- Без излишней структурированности - просто нормальный текст
- Дружелюбный, но профессиональный тон
- Не упоминай, что ты AI или бот
- Пиши абзацами, разделяй мысли пустыми строками

ФОРМАТ ТЕКСТА:
- НЕ используй жирный текст, курсив, заголовки
- НЕ используй маркированные или нумерованные списки
- Пиши просто обычным текстом, как в WhatsApp или Telegram
- Если нужно перечислить - просто пиши "во-первых", "во-вторых" или через запятую
- Разделяй абзацы пустой строкой для читаемости

СТРУКТУРА ОТВЕТА:
Сначала дай прямой ответ на вопрос, потом детали и объяснения. Пиши так, будто набираешь на телефоне - естественно и без вычурности.

ОЦЕНКА УВЕРЕННОСТИ:
В конце КАЖДОГО ответа ОБЯЗАТЕЛЬНО добавь строку:
CONFIDENCE: [число от 0.0 до 1.0]

Критерии уверенности:
- 0.9-1.0: Точная информация из базы знаний, полностью уверен
- 0.7-0.8: Хорошее понимание темы, общая информация корректна
- 0.5-0.6: Частичная уверенность, информация может требовать уточнений
- 0.3-0.4: Низкая уверенность, нужна консультация с коллегами
- 0.0-0.2: Не уверен, требуется помощь эксперта

ВАЖНО: 
- Всегда отвечай на русском языке
- Базируйся на информации из базы знаний
- Если информации недостаточно, честно признай это и поставь низкую уверенность
- Не придумывай факты, лучше признать незнание
- НИКАКОГО MARKDOWN - только простой текст!"""

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
        
        user_prompt += "Дай ответ от имени Сергея с оценкой уверенности в конце."
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
                top_p=1
            )
            
            content = response.choices[0].message.content
            
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
                reasoning=f"OpenAI API ({self.model}), context items: {len(context) if context else 0}"
            )
            
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
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
        Генерирует эмбеддинг через OpenAI API
        Использует text-embedding-3-small - недорогая и качественная модель
        """
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float"
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            print(f"❌ OpenAI Embedding error: {e}")
            try:
                from sentence_transformers import SentenceTransformer
                
                if not hasattr(self, '_embedding_model'):
                    print("🔄 Fallback: загрузка локальной embedding модели...")
                    self._embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                
                embedding = self._embedding_model.encode(text)
                return embedding.tolist()
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed for fallback. "
                    "Install it with: pip install sentence-transformers"
                )