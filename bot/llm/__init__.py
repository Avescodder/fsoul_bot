import os
from bot.llm.base import BaseLLM
from bot.llm.ollama import OllamaLLM
from bot.llm.claude import ClaudeLLM
from bot.llm.groq import GroqLLM
from bot.llm.openai import OpenAILLM


def get_llm() -> BaseLLM:
    """Factory для получения LLM провайдера"""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if provider == "openai":
        return OpenAILLM()
    elif provider == "groq":
        return GroqLLM()
    elif provider == "claude":
        return ClaudeLLM()
    elif provider == "ollama":
        return OllamaLLM()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")