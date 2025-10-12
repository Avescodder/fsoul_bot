"""
Скрипт для тестирования LLM провайдера
Запуск: python test_llm.py
"""

import asyncio
import os
from dotenv import load_dotenv
from bot.llm import get_llm

load_dotenv()


async def test_basic_question():
    """Тест простого вопроса"""
    print("=" * 60)
    print("ТЕСТ 1: Простой вопрос")
    print("=" * 60)
    
    llm = get_llm()
    question = "Какие документы нужны для визы D7?"
    
    print(f"\n❓ Вопрос: {question}")
    print("\n⏳ Генерирую ответ...\n")
    
    response = await llm.generate_answer(question)
    
    print(f"✅ Ответ:\n{response.answer}")
    print(f"\n📊 Уверенность: {response.confidence:.2%}")
    print(f"🔍 Reasoning: {response.reasoning}")


async def test_with_context():
    """Тест с контекстом из базы знаний"""
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Вопрос с контекстом")
    print("=" * 60)
    
    llm = get_llm()
    question = "Сколько стоит виза D7?"
    
    # Имитируем контекст из базы знаний
    context = [
        (
            "Какие расходы на оформление визы D7?",
            "Консульский сбор - €83. Также нужно учесть перевод документов (~€200), апостиль (€10-30 за документ)."
        ),
        (
            "Сколько времени занимает получение D7?",
            "Обычно 2-3 месяца от подачи до получения визы."
        )
    ]
    
    print(f"\n❓ Вопрос: {question}")
    print(f"\n📚 Контекст из базы знаний ({len(context)} записей)")
    print("\n⏳ Генерирую ответ...\n")
    
    response = await llm.generate_answer(question, context)
    
    print(f"✅ Ответ:\n{response.answer}")
    print(f"\n📊 Уверенность: {response.confidence:.2%}")
    print(f"🔍 Reasoning: {response.reasoning}")


async def test_embedding():
    """Тест генерации эмбеддингов"""
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Генерация эмбеддингов")
    print("=" * 60)
    
    llm = get_llm()
    text = "Как получить визу D7 в Португалию?"
    
    print(f"\n📝 Текст: {text}")
    print("\n⏳ Генерирую эмбеддинг...\n")
    
    embedding = await llm.generate_embedding(text)
    
    print(f"✅ Эмбеддинг сгенерирован")
    print(f"📊 Размерность: {len(embedding)}")
    print(f"🔢 Первые 5 значений: {embedding[:5]}")


async def test_confidence_levels():
    """Тест разных уровней уверенности"""
    print("\n" + "=" * 60)
    print("ТЕСТ 4: Уровни уверенности")
    print("=" * 60)
    
    llm = get_llm()
    
    questions = [
        ("Какая столица Португалии?", "Простой факт"),
        ("Какие документы нужны для D7?", "Стандартный вопрос"),
        ("Могу ли я получить визу если у меня судимость за неуплату налогов 10 лет назад?", "Сложный кейс")
    ]
    
    for question, description in questions:
        print(f"\n{'─' * 60}")
        print(f"📌 {description}")
        print(f"❓ {question}")
        
        response = await llm.generate_answer(question)
        
        print(f"📊 Уверенность: {response.confidence:.2%}")
        
        threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
        if response.confidence >= threshold:
            print("✅ Будет отвечено автоматически")
        else:
            print("⚠️  Будет эскалировано админам")


async def test_multilingual():
    """Тест на разных языках"""
    print("\n" + "=" * 60)
    print("ТЕСТ 5: Мультиязычность")
    print("=" * 60)
    
    llm = get_llm()
    
    questions = [
        ("Quais são os requisitos para o visto D7?", "Португальский"),
        ("What are the requirements for D7 visa?", "Английский"),
        ("Какие требования для визы D7?", "Русский")
    ]
    
    for question, lang in questions:
        print(f"\n{'─' * 60}")
        print(f"🌍 Язык: {lang}")
        print(f"❓ {question}")
        print("\n⏳ Генерирую ответ...\n")
        
        response = await llm.generate_answer(question)
        
        print(f"✅ Ответ:\n{response.answer[:200]}...")
        print(f"\n📊 Уверенность: {response.confidence:.2%}")


async def main():
    """Запуск всех тестов"""
    print("\n🧪 ТЕСТИРОВАНИЕ LLM ПРОВАЙДЕРА")
    print(f"🔧 Provider: {os.getenv('LLM_PROVIDER', 'ollama')}")
    print(f"🤖 Model: {os.getenv('OLLAMA_MODEL', 'N/A')}")
    print(f"🎯 Threshold: {os.getenv('CONFIDENCE_THRESHOLD', '0.7')}")
    
    try:
        await test_basic_question()
        await test_with_context()
        await test_embedding()
        await test_confidence_levels()
        await test_multilingual()
        
        print("\n" + "=" * 60)
        print("✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        print("\nПроверь:")
        print("  - Запущен ли Ollama: docker-compose ps")
        print("  - Загружена ли модель: docker exec <ollama-container> ollama list")
        print("  - Правильно ли настроен .env файл")


if __name__ == "__main__":
    asyncio.run(main())