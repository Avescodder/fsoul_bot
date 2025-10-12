"""
Утилиты для управления базой данных
"""

import asyncio
from database import get_db, init_db
from database.models import User, Question, KnowledgeBase, PendingQuestion
from bot.llm import get_llm
from utils.rag import RAGSystem


async def seed_knowledge_base():
    """Заполняет базу знаний начальными данными"""
    print("🌱 Заполняю базу знаний начальными данными...")
    
    llm = get_llm()
    rag = RAGSystem(llm)
    
    initial_knowledge = [
        (
            "Какие документы нужны для визы D7?",
            "Для визы D7 необходимы: действующий паспорт, фотографии 3.5x4.5 см, "
            "подтверждение дохода (минимум €760/месяц), справка о несудимости, "
            "медицинская страховка, подтверждение проживания в Португалии."
        ),
        (
            "Сколько времени занимает получение визы D7?",
            "Процесс получения визы D7 обычно занимает 2-3 месяца от подачи документов "
            "до получения визы. Время может варьироваться в зависимости от консульства."
        ),
        (
            "Что такое NIF и как его получить?",
            "NIF (Número de Identificação Fiscal) - это налоговый номер в Португалии. "
            "Получить можно в финансовой инспекции (Finanças) или через представителя. "
            "Обычно выдается в течение 1-2 рабочих дней."
        ),
        (
            "Какой минимальный доход нужен для D7?",
            "Минимальный пассивный доход для визы D7: €760/месяц на основного заявителя, "
            "+50% (€380) на супруга, +30% (€228) на каждого ребенка или иждивенца."
        ),
        (
            "Можно ли работать на визе D7?",
            "Да, виза D7 позволяет работать в Португалии. Однако основное требование - "
            "наличие пассивного дохода. Работа должна быть дополнением, а не основным источником."
        ),
        (
            "Какие налоги в Португалии для резидентов?",
            "Прогрессивная шкала подоходного налога: от 14.5% до 48%. Возможна регистрация "
            "в режиме NHR (Non-Habitual Resident) с льготными ставками на 10 лет."
        ),
        (
            "Что такое режим NHR?",
            "NHR (Non-Habitual Resident) - налоговый режим для новых резидентов Португалии. "
            "Позволяет платить 0-20% налога на иностранные доходы в течение 10 лет."
        ),
        (
            "Нужно ли знать португальский для D7?",
            "Для получения визы D7 знание португальского языка не требуется. Однако для "
            "получения ВНЖ и гражданства потребуется сдать экзамен на уровень A2."
        )
    ]
    
    with get_db() as db:
        for question, answer in initial_knowledge:
            try:
                await rag.add_to_knowledge_base(
                    db,
                    question=question,
                    answer=answer,
                    source="manual",
                    verified=True
                )
                print(f"  ✅ Добавлено: {question[:50]}...")
            except Exception as e:
                print(f"  ❌ Ошибка: {e}")
    
    print(f"✅ Добавлено {len(initial_knowledge)} записей в базу знаний")


def show_stats():
    """Показывает статистику базы данных"""
    print("\n📊 СТАТИСТИКА БАЗЫ ДАННЫХ")
    print("=" * 60)
    
    with get_db() as db:
        users_count = db.query(User).count()
        questions_count = db.query(Question).count()
        kb_count = db.query(KnowledgeBase).count()
        pending_count = db.query(PendingQuestion).count()
        
        answered_by_ai = db.query(Question).filter(
            Question.answered_by_ai == True
        ).count()
        
        answered_by_admin = db.query(Question).filter(
            Question.answered_by_ai == False,
            Question.status == "answered"
        ).count()
        
        print(f"\n👥 Пользователей: {users_count}")
        print(f"❓ Всего вопросов: {questions_count}")
        print(f"   ├─ Ответов от AI: {answered_by_ai}")
        print(f"   ├─ Ответов от админов: {answered_by_admin}")
        print(f"   └─ В ожидании: {pending_count}")
        print(f"📚 База знаний: {kb_count} записей")
        
        if questions_count > 0:
            ai_rate = (answered_by_ai / questions_count) * 100
            print(f"\n🤖 AI Resolution Rate: {ai_rate:.1f}%")
        
        from sqlalchemy import func
        avg_confidence = db.query(func.avg(Question.confidence_score)).scalar()
        if avg_confidence:
            print(f"📊 Средняя уверенность: {avg_confidence:.2%}")
        
        from sqlalchemy import desc
        top_users = db.query(
            User.first_name,
            User.username,
            func.count(Question.id).label('count')
        ).join(Question).group_by(User.id).order_by(desc('count')).limit(5).all()
        
        if top_users:
            print(f"\n👑 Топ-5 активных пользователей:")
            for i, (name, username, count) in enumerate(top_users, 1):
                print(f"   {i}. {name or username or 'Аноним'}: {count} вопросов")


def clear_database():
    """Очищает все данные из базы (осторожно!)"""
    print("\n⚠️  ВНИМАНИЕ: Это удалит ВСЕ данные из базы!")
    confirm = input("Введи 'DELETE' для подтверждения: ")
    
    if confirm != "DELETE":
        print("❌ Отменено")
        return
    
    with get_db() as db:
        db.query(PendingQuestion).delete()
        db.query(Question).delete()
        db.query(KnowledgeBase).delete()
        db.query(User).delete()
        db.commit()
    
    print("✅ База данных очищена")


def export_knowledge_base(filename: str = "knowledge_base_export.txt"):
    """Экспортирует базу знаний в текстовый файл"""
    print(f"\n📤 Экспорт базы знаний в {filename}...")
    
    with get_db() as db:
        kb_entries = db.query(KnowledgeBase).filter(
            KnowledgeBase.verified == True
        ).order_by(KnowledgeBase.usage_count.desc()).all()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# БАЗА ЗНАНИЙ IMMIGRATION BOT\n")
            f.write(f"# Всего записей: {len(kb_entries)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, entry in enumerate(kb_entries, 1):
                f.write(f"## Запись #{i}\n")
                f.write(f"Вопрос: {entry.question}\n\n")
                f.write(f"Ответ: {entry.answer}\n\n")
                f.write(f"Источник: {entry.source}\n")
                f.write(f"Использований: {entry.usage_count}\n")
                f.write(f"Создано: {entry.created_at}\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"✅ Экспортировано {len(kb_entries)} записей")


async def test_rag_search(query: str):
    """Тестирует RAG поиск"""
    print(f"\n🔍 Поиск похожих вопросов для: '{query}'")
    print("=" * 60)
    
    llm = get_llm()
    rag = RAGSystem(llm)
    
    with get_db() as db:
        similar = await rag.search_similar(db, query)
        
        if not similar:
            print("❌ Похожих вопросов не найдено")
            return
        
        print(f"\n✅ Найдено {len(similar)} похожих вопросов:\n")
        
        for i, (q, a) in enumerate(similar, 1):
            print(f"{i}. Вопрос: {q}")
            print(f"   Ответ: {a[:100]}...")
            print()


def main_menu():
    """Главное меню утилиты"""
    while True:
        print("\n" + "=" * 60)
        print("🛠️  DATABASE MANAGER")
        print("=" * 60)
        print("1. Показать статистику")
        print("2. Заполнить базу знаний начальными данными")
        print("3. Экспортировать базу знаний")
        print("4. Тест RAG поиска")
        print("5. Очистить базу данных (⚠️  опасно)")
        print("0. Выход")
        
        choice = input("\nВыбери опцию: ")
        
        if choice == "1":
            show_stats()
        elif choice == "2":
            asyncio.run(seed_knowledge_base())
        elif choice == "3":
            filename = input("Имя файла (Enter для knowledge_base_export.txt): ").strip()
            export_knowledge_base(filename or "knowledge_base_export.txt")
        elif choice == "4":
            query = input("Введи вопрос для поиска: ").strip()
            if query:
                asyncio.run(test_rag_search(query))
        elif choice == "5":
            clear_database()
        elif choice == "0":
            print("👋 До встречи!")
            break
        else:
            print("❌ Неверный выбор")


if __name__ == "__main__":
    # Инициализация БД
    init_db()
    main_menu()