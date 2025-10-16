from typing import List, Tuple
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from database.models import KnowledgeBase
from bot.llm.base import BaseLLM


class RAGSystem:
    """Retrieval-Augmented Generation система"""
    
    def __init__(self, llm: BaseLLM, top_k: int = 3):
        self.llm = llm
        self.top_k = top_k
    
    async def search_similar(
        self, 
        db: Session, 
        question: str
    ) -> List[Tuple[str, str]]:
        """
        Ищет похожие вопросы в базе знаний
        
        Returns:
            List[(question, answer)]
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
        
        similar = [(r.question, r.answer) for r in results if r.distance < 0.5]
        
        return similar
    
    async def get_answer(
        self, 
        db: Session, 
        question: str
    ) -> Tuple[str, float, List[Tuple[str, str]]]:
        """
        Получает ответ с использованием RAG
        
        Returns:
            (answer, confidence, context)
        """
        context = await self.search_similar(db, question)
        
        response = await self.llm.generate_answer(question, context)
        
        return response.answer, response.confidence, context
    
    async def add_to_knowledge_base(
        self,
        db: Session,
        question: str,
        answer: str,
        source: str = "admin",
        verified: bool = True
    ):
        """Добавляет новую пару Q&A в базу знаний"""
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