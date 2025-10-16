from typing import List, Tuple, Optional
from sqlalchemy import select, func, desc
from sqlalchemy.orm import Session
from database.models import KnowledgeBase, Question, User
from bot.llm.base import BaseLLM
from datetime import datetime, timedelta


class ImprovedRAGSystem:
    """Улучшенная RAG система с контекстом и веб-поиском"""
    
    def __init__(self, llm: BaseLLM, top_k: int = 5):
        self.llm = llm
        self.top_k = top_k
        self.conversation_cache = {}
    
    async def search_similar(
        self, 
        db: Session, 
        question: str,
        threshold: float = 0.4  
    ) -> List[Tuple[str, str, float]]:
        """
        Ищет похожие вопросы в базе знаний
        
        Returns:
            List[(question, answer, similarity_score)]
        """
        question_embedding = await self.llm.generate_embedding(question)
        
        results = db.execute(
            select(
                KnowledgeBase.question,
                KnowledgeBase.answer,
                KnowledgeBase.question_embedding.cosine_distance(question_embedding).label('distance')
            )
            .where(KnowledgeBase.verified == True)
            .order_by('distance')
            .limit(self.top_k)
        ).fetchall()
        
        similar = [
            (r.question, r.answer, 1 - r.distance) 
            for r in results 
            if r.distance < threshold
        ]
        
        return similar
    
    def get_conversation_history(
        self, 
        db: Session, 
        user_id: int, 
        limit: int = 5
    ) -> List[Tuple[str, str]]:
        """Получает историю последних диалогов пользователя"""
        
        questions = db.query(Question).filter(
            Question.user_id == user_id,
            Question.status == "answered",
            Question.created_at > datetime.utcnow() - timedelta(hours=24)
        ).order_by(desc(Question.created_at)).limit(limit).all()
        
        history = []
        for q in reversed(questions):
            if q.question_text and q.answer_text:
                history.append((q.question_text, q.answer_text))
        
        return history
    
    def is_greeting(self, question: str) -> bool:
        """Проверяет является ли вопрос приветствием"""
        greetings = [
            'привет', 'здравствуй', 'добрый день', 'добрый вечер', 'добрый утро',
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            'ola', 'bom dia', 'boa tarde', 'boa noite',
            'как дела', 'how are you', 'como está'
        ]
        
        q_lower = question.lower().strip()
        return any(g in q_lower for g in greetings) and len(q_lower.split()) < 5
    
    def is_simple_question(self, question: str) -> bool:
        """Определяет простые вопросы которые не требуют эскалации"""
        simple_patterns = [
            'спасибо', 'thanks', 'obrigado',
            'понятно', 'ясно', 'got it', 'ok', 'okay',
            'да', 'нет', 'yes', 'no', 'sim', 'não'
        ]
        
        q_lower = question.lower().strip()
        return any(p in q_lower for p in simple_patterns) and len(question.split()) < 10
    
    async def get_answer(
        self, 
        db: Session, 
        question: str,
        user_id: int,
        use_web_search: bool = False
    ) -> Tuple[str, float, List[Tuple[str, str]]]:
        """
        Получает ответ с использованием RAG + контекст + веб-поиск
        
        Returns:
            (answer, confidence, context)
        """
        
        if self.is_greeting(question):
            greeting_response = (
                "Здравствуйте! Рад помочь вам с вопросами по иммиграции в Португалию. "
                "Задавайте ваши вопросы, постараюсь ответить максимально подробно."
            )
            return greeting_response, 1.0, []
        
        if self.is_simple_question(question):
            simple_response = "Рад быть полезным! Если возникнут ещё вопросы - обращайтесь."
            return simple_response, 1.0, []
        
        kb_context = await self.search_similar(db, question)
        
        conversation_history = self.get_conversation_history(db, user_id, limit=3)
        
        # Веб-поиск если нужно (TODO: интеграция)
        web_context = None
        if use_web_search and not kb_context:
            web_context = await self._search_web(question)
        
        response = await self.llm.generate_answer(
            question=question,
            context=[(q, a) for q, a, _ in kb_context],
            conversation_history=conversation_history,
            web_context=web_context
        )
        
        if kb_context and kb_context[0][2] > 0.8:  # similarity > 80%
            response.confidence = max(response.confidence, 0.85)
        
        return response.answer, response.confidence, [(q, a) for q, a, _ in kb_context]
    
    async def _search_web(self, query: str) -> Optional[str]:
        """Поиск в интернете (заглушка для будущей интеграции)"""
        # TODO: Интегрировать с Tavily, SerpAPI или другим поисковиком
        return None
    
    async def add_to_knowledge_base(
        self,
        db: Session,
        question: str,
        answer: str,
        source: str = "admin",
        verified: bool = True
    ):
        """Добавляет новую пару Q&A в базу знаний"""
        
        existing = db.query(KnowledgeBase).filter(
            KnowledgeBase.question == question
        ).first()
        
        if existing:
            existing.answer = answer
            existing.source = source
            existing.verified = verified
            db.commit()
            return existing
        
        embedding = await self.llm.generate_embedding(question)
        
        kb_entry = KnowledgeBase(
            question=question,
            answer=answer,
            question_embedding=embedding,
            source=source,
            verified=verified
        )
        
        db.add(kb_entry)
        db.commit()
        
        return kb_entry