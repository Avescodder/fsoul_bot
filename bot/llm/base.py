from abc import ABC, abstractmethod
from typing import Tuple, List
from pydantic import BaseModel


class LLMResponse(BaseModel):
    answer: str
    confidence: float  # 0.0 - 1.0
    reasoning: str = ""


class BaseLLM(ABC):
    """Абстрактный класс для LLM провайдеров"""
    
    @abstractmethod
    async def generate_answer(
        self, 
        question: str, 
        context: List[Tuple[str, str]] = None
    ) -> LLMResponse:
        """
        Генерирует ответ на вопрос
        
        Args:
            question: Вопрос пользователя
            context: Список (вопрос, ответ) из базы знаний
            
        Returns:
            LLMResponse с ответом и уверенностью
        """
        pass
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Генерирует эмбеддинг для текста
        
        Args:
            text: Текст для эмбеддинга
            
        Returns:
            Вектор эмбеддинга
        """
        pass