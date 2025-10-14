import os
from bot.llm.base import BaseLLM
from bot.llm.groq import GroqLLM
from bot.llm.openai import OpenAILLM


def get_llm() -> BaseLLM:
    """Factory для получения LLM провайдера"""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if provider == "openai":
        return OpenAILLM()
    elif provider == "groq":
        return GroqLLM()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")