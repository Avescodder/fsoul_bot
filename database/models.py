import os
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, ForeignKey, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "local": 768,  
}

if LLM_PROVIDER == "openai":
    VECTOR_DIM = EMBEDDING_DIMENSIONS.get(EMBEDDING_MODEL, 1536)
else:
    VECTOR_DIM = 768  


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, unique=True, nullable=False, index=True)
    username = Column(String(255))
    first_name = Column(String(255))
    last_name = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    questions = relationship("Question", back_populates="user")


class Question(Base):
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    message_id = Column(BigInteger)
    question_text = Column(Text, nullable=False)
    question_embedding = Column(Vector(VECTOR_DIM))
    answer_text = Column(Text)
    confidence_score = Column(Float)
    answered_by_ai = Column(Boolean, default=True)
    answered_by_admin_id = Column(BigInteger)
    status = Column(String(50), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    answered_at = Column(DateTime)
    
    user = relationship("User", back_populates="questions")


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    
    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    question_embedding = Column(Vector(VECTOR_DIM))
    source = Column(String(255))
    verified = Column(Boolean, default=False)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PendingQuestion(Base):
    __tablename__ = "pending_questions"
    
    id = Column(Integer, primary_key=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    user_telegram_id = Column(BigInteger, nullable=False)
    forwarded_to_admins = Column(Boolean, default=False)
    admin_message_ids = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)