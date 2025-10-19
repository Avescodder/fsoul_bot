import os
import httpx
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TavilyWebSearch:
    """Интеграция с Tavily API для веб-поиска"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com/search"
        self.timeout = 10.0
        
        if not self.api_key:
            logger.warning("⚠️ TAVILY_API_KEY не установлен. Веб-поиск отключен.")
    
    async def search(
        self, 
        query: str,
        max_results: int = 5,
        include_answer: bool = True,
        search_depth: str = "basic",  
        topic: str = "general" 
    ) -> Optional[Dict[str, Any]]:
        """
        Выполняет веб-поиск через Tavily
        
        Args:
            query: поисковый запрос
            max_results: максимум результатов (1-20)
            include_answer: включить готовый ответ из Tavily
            search_depth: глубина поиска
            topic: категория поиска ("general" или "news")
        
        Returns:
            Словарь с результатами или None в случае ошибки
        """
        if not self.api_key:
            logger.error("❌ API ключ Tavily не установлен")
            return None
        
        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": min(max_results, 20),
                "include_answer": include_answer,
                "search_depth": search_depth,
                "topic": topic,
                "include_images": False,  
                "include_raw_content": True 
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                logger.info(f"✅ Tavily поиск успешен: {query}")
                return data
        
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP ошибка Tavily ({e.status_code}): {e.response.text}")
            return None
        except httpx.TimeoutException:
            logger.error(f"⏱️ Timeout при запросе к Tavily")
            return None
        except httpx.RequestError as e:
            logger.error(f"❌ Ошибка запроса Tavily: {e}")
            return None
    
    async def format_results(
        self, 
        search_data: Dict[str, Any],
        max_sources: int = 5
    ) -> str:
        """
        Форматирует результаты поиска в читаемый текст
        
        Args:
            search_data: результаты от Tavily
            max_sources: максимум источников в выводе
        
        Returns:
            Отформатированная строка с результатами
        """
        if not search_data:
            return ""
        
        result = ""
        
        if search_data.get("answer"):
            result += f"📌 **Основной ответ:**\n{search_data['answer']}\n\n"
        
        results = search_data.get("results", [])
        if results:
            result += f"📚 **Топ-{min(len(results), max_sources)} источников:**\n"
            
            for idx, item in enumerate(results[:max_sources], 1):
                title = item.get("title", "Без названия")
                content = item.get("content", "Нет описания")
                url = item.get("url", "")
                
                result += f"\n{idx}. **{title}**\n"
                result += f"   {content[:200]}...\n"
                if url:
                    result += f"   🔗 {url}\n"
        
        return result.strip()

class ImprovedRAGSystemWithTavily:
    """Расширенная RAG система с Tavily веб-поиском"""
    
    def __init__(self, llm, top_k: int = 5, tavily_api_key: Optional[str] = None):
        self.llm = llm
        self.top_k = top_k
        self.conversation_cache = {}
        self.web_search = TavilyWebSearch(api_key=tavily_api_key)
        
        self.search_cache = {}
        self.cache_ttl = timedelta(hours=1)
    
    async def _search_web(
        self, 
        query: str,
        use_cache: bool = True,
        search_depth: str = "basic"
    ) -> Optional[str]:
        """
        Поиск в интернете с поддержкой кеша
        
        Args:
            query: поисковый запрос
            use_cache: использовать кеш результатов
            search_depth: "basic" или "advanced" (advanced дороже)
        
        Returns:
            Отформатированные результаты поиска или None
        """
        
        if use_cache and query in self.search_cache:
            cached_time, cached_result = self.search_cache[query]
            if datetime.utcnow() - cached_time < self.cache_ttl:
                logger.info(f"📦 Результат из кеша для: {query}")
                return cached_result
        
        search_data = await self.web_search.search(
            query=query,
            max_results=5,
            include_answer=True,
            search_depth=search_depth,
            topic="general"
        )
        
        if not search_data:
            return None
        
        formatted = await self.web_search.format_results(search_data, max_sources=5)
        
        if use_cache:
            self.search_cache[query] = (datetime.utcnow(), formatted)
        
        return formatted
    
    async def get_answer_with_web_search(
        self,
        db,
        question: str,
        user_id: int,
        use_web_search: bool = False,
        search_depth: str = "basic"
    ) -> tuple:
        """
        Получает ответ с использованием RAG + контекст + веб-поиск
        
        Returns:
            (answer, confidence, context_sources)
        """
        
        kb_context = await self.search_similar(db, question)
        
        conversation_history = self.get_conversation_history(db, user_id, limit=3)
        
        web_context = None
        if use_web_search or (not kb_context and not self.is_simple_question(question)):
            web_context = await self._search_web(
                question, 
                use_cache=True,
                search_depth=search_depth
            )
            if web_context:
                logger.info(f"🌐 Добавлен веб-контекст для вопроса: {question}")
        
        response = await self.llm.generate_answer(
            question=question,
            context=[(q, a) for q, a, _ in kb_context],
            conversation_history=conversation_history,
            web_context=web_context
        )
        
        if kb_context and kb_context[0][2] > 0.8:
            response.confidence = max(response.confidence, 0.85)
        elif web_context:
            response.confidence = max(response.confidence, 0.75)
        
        sources = [(q, a) for q, a, _ in kb_context]
        if web_context:
            sources.append(("Web Search", web_context))
        
        return response.answer, response.confidence, sources
    
    def is_simple_question(self, question: str) -> bool:
        """Определяет простые вопросы"""
        simple_patterns = [
            'спасибо', 'thanks', 'obrigado',
            'понятно', 'ясно', 'got it', 'ok',
            'да', 'нет', 'yes', 'no'
        ]
        q_lower = question.lower().strip()
        return any(p in q_lower for p in simple_patterns) and len(question.split()) < 10
    
    def clear_cache(self):
        """Очищает кеш поиска"""
        self.search_cache.clear()
        logger.info("🗑️ Кеш поиска очищен")

