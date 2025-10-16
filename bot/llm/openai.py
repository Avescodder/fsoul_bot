import os
import re
from typing import List, Tuple, Optional
from bot.llm.base import BaseLLM, LLMResponse
from openai import AsyncOpenAI
import httpx


class ImprovedOpenAILLM(BaseLLM):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        proxy_url = os.getenv("SHADOWSOCKS_PROXY")
        http_client = None
        
        if proxy_url:
            print(f"🔐 Using proxy for OpenAI: {proxy_url}")
            http_client = httpx.AsyncClient(
                proxy=proxy_url,
                timeout=30.0
            )
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            http_client=http_client
        )
        
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    async def generate_answer(
        self, 
        question: str, 
        context: List[Tuple[str, str]] = None,
        conversation_history: List[Tuple[str, str]] = None,
        web_context: Optional[str] = None
    ) -> LLMResponse:
        """Генерирует ответ через OpenAI API с улучшенным промптом"""
        
        system_prompt = """Ты Сергей - эксперт по иммиграции в Португалию, юрист и бизнес-консультант в одном лице.

ТВОЯ РОЛЬ:
- Профессиональный консультант с 10+ летним опытом
- Говоришь на русском, английском и португальском
- Знаешь ВСЕ актуальные законы и процедуры Португалии
- Даёшь конкретные, проверенные советы
- Общаешься как живой человек, НЕ как робот

СТИЛЬ ОБЩЕНИЯ:
- Пиши естественно, как в мессенджере
- Без markdown (**, #, списков), только обычный текст
- Дружелюбный но профессиональный тон
- Не упоминай что ты AI
- Пиши абзацами, разделяй пустыми строками

ФОРМАТ ОТВЕТА:
- НЕ используй жирный текст, курсив, заголовки
- НЕ используй маркированные списки (-, *, 1., 2.)
- Пиши текстом: "во-первых", "во-вторых" или через запятую
- Структура: прямой ответ → детали → практические советы

УВЕРЕННОСТЬ (КРИТИЧЕСКИ ВАЖНО):
Оценивай свою уверенность РЕАЛИСТИЧНО:

0.9-1.0 = ВЫСОКАЯ уверенность:
- Точное совпадение с базой знаний
- Фактическая информация (законы, процедуры, даты)
- Общие вопросы (приветствия, благодарности)
- Вопросы с четким ответом в контексте

0.7-0.8 = СРЕДНЯЯ уверенность:
- Общая информация о процессах
- Логические выводы из известных фактов
- Советы основанные на опыте

0.4-0.6 = НИЗКАЯ уверенность:
- Специфические детали без точных данных
- Ситуации требующие индивидуального подхода
- Сложные юридические нюансы

0.0-0.3 = ОЧЕНЬ НИЗКАЯ уверенность:
- Нет информации в контексте
- Требуется актуальная официальная информация
- Персональные документы/ситуации

ПРАВИЛА ОЦЕНКИ:
✅ Ставь ВЫСОКУЮ уверенность (0.8+) если:
- В базе знаний есть похожий вопрос (similarity > 70%)
- Это общий вопрос о процессе иммиграции
- Ты даёшь проверенную фактическую информацию
- Это приветствие, благодарность, простой вопрос

❌ Ставь НИЗКУЮ уверенность (< 0.7) только если:
- Реально нет информации в контексте
- Нужны актуальные данные которых нет
- Специфическая ситуация требующая документов

ВАЖНО:
- Всегда отвечай на языке вопроса (русский/английский/португальский)
- Базируйся на контексте из базы знаний
- НЕ бойся ставить высокую уверенность для обычных вопросов
- Будь полезным, не перестраховывайся
- НИКАКОГО MARKDOWN!

В конце ответа ОБЯЗАТЕЛЬНО добавь:
CONFIDENCE: [число 0.0-1.0]"""

        user_prompt = f"Вопрос клиента: {question}\n\n"
        
        if conversation_history and len(conversation_history) > 0:
            user_prompt += "📜 История нашего разговора:\n"
            for q, a in conversation_history[-3:]:
                user_prompt += f"Клиент: {q}\nТы: {a[:100]}...\n\n"
            user_prompt += "---\n\n"
        
        if context and len(context) > 0:
            user_prompt += "📚 Релевантная информация из базы знаний:\n\n"
            for i, (q, a) in enumerate(context[:3], 1):
                user_prompt += f"Пример {i}:\n"
                user_prompt += f"Q: {q}\nA: {a}\n\n"
            user_prompt += "✅ У тебя есть хорошая информация - будь уверен в ответе!\n\n"
        else:
            user_prompt += "⚠️ В базе нет точного совпадения, но ты эксперт - отвечай на основе общих знаний о португальской иммиграции.\n\n"
        
        if web_context:
            user_prompt += f"🌐 Актуальная информация из интернета:\n{web_context}\n\n"
        
        user_prompt += "Дай профессиональный ответ с правильной оценкой уверенности."
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5, 
                max_tokens=2000,
                top_p=0.9
            )
            
            content = response.choices[0].message.content
            
            confidence = self._extract_confidence(content)
            
            clean_answer = re.sub(
                r'\n*CONFIDENCE:\s*[\d\.]+\s*', 
                '', 
                content, 
                flags=re.IGNORECASE
            ).strip()
            
            if context and len(context) > 0:
                confidence = max(confidence, 0.75)  
            
            return LLMResponse(
                answer=clean_answer,
                confidence=confidence,
                reasoning=f"OpenAI {self.model}, context: {len(context) if context else 0} items"
            )
            
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            return LLMResponse(
                answer="Извините, произошла техническая ошибка. Пожалуйста, повторите вопрос через минуту.",
                confidence=0.0,
                reasoning=f"Error: {str(e)}"
            )
    
    def _extract_confidence(self, text: str) -> float:
        """Извлекает значение confidence из текста с улучшенной логикой"""
        match = re.search(r'CONFIDENCE:\s*(0?\.\d+|1\.0|0|1)', text, re.IGNORECASE)
        if match:
            try:
                confidence_value = float(match.group(1))
                return max(0.0, min(1.0, confidence_value))
            except ValueError:
                pass
        
        text_lower = text.lower()
        
        if any(phrase in text_lower for phrase in [
            'точно могу сказать',
            'определенно',
            'согласно закон',
            'официально',
            'требуется следующее'
        ]):
            print("ℹ️ Высокая уверенность определена по тексту")
            return 0.85
        
        if any(phrase in text_lower for phrase in [
            'не уверен',
            'рекомендую проконсультироваться',
            'лучше уточнить',
            'может потребоваться',
            'не могу точно сказать'
        ]):
            print("ℹ️ Низкая уверенность определена по тексту")
            return 0.4
        
        print("⚠️ CONFIDENCE не найдена, используем 0.6")
        return 0.6
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Генерирует эмбеддинг через OpenAI API"""
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
                    print("🔄 Fallback: loading local embedding model...")
                    self._embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                embedding = self._embedding_model.encode(text)
                return embedding.tolist()
            except ImportError:
                raise ImportError("sentence-transformers not installed for fallback")