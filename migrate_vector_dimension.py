"""
Скрипт миграции для изменения размерности векторов
Использовать если база уже существует с другой размерностью
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

TARGET_DIM = 1536 if LLM_PROVIDER == "openai" else 768

print(f"🔧 Миграция векторов на размерность: {TARGET_DIM}")
print(f"📊 LLM Provider: {LLM_PROVIDER}")
print("=" * 60)

engine = create_engine(DATABASE_URL)

def check_current_dimension(table_name: str, column_name: str):
    """Проверяет текущую размерность вектора"""
    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT atttypmod 
            FROM pg_attribute 
            WHERE attrelid = '{table_name}'::regclass 
            AND attname = '{column_name}'
        """))
        row = result.fetchone()
        if row and row[0] > 0:
            return row[0]
        return None

def migrate_table(table_name: str, column_name: str):
    """Мигрирует таблицу на новую размерность"""
    current_dim = check_current_dimension(table_name, column_name)
    
    print(f"\n📋 Таблица: {table_name}.{column_name}")
    print(f"   Текущая размерность: {current_dim or 'не определена'}")
    print(f"   Целевая размерность: {TARGET_DIM}")
    
    if current_dim == TARGET_DIM:
        print(f"   ✅ Уже правильная размерность, пропускаем")
        return
    
    with engine.begin() as conn:
        print(f"   🗑️  Удаляем старую колонку...")
        conn.execute(text(f"""
            ALTER TABLE {table_name} 
            DROP COLUMN IF EXISTS {column_name}
        """))
        
        print(f"   ➕ Создаем новую колонку vector({TARGET_DIM})...")
        conn.execute(text(f"""
            ALTER TABLE {table_name} 
            ADD COLUMN {column_name} vector({TARGET_DIM})
        """))
        
        print(f"   ✅ Готово!")

def main():
    try:
        print("\n⚠️  ВНИМАНИЕ: Это удалит все существующие эмбеддинги!")
        print("   Они будут пересозданы при следующем запуске бота.")
        
        confirm = input("\nПродолжить? (yes/no): ").strip().lower()
        
        if confirm != 'yes':
            print("❌ Отменено")
            return
        
        print("\n🚀 Начинаю миграцию...\n")
        
        migrate_table('questions', 'question_embedding')
        migrate_table('knowledge_base', 'question_embedding')
        
        print("\n" + "=" * 60)
        print("✅ Миграция завершена успешно!")
        print("\n📝 Следующие шаги:")
        print("   1. Перезапусти бота: docker-compose restart bot")
        print("   2. Перезаполни базу знаний: docker-compose run --rm bot python -m utils.db_manager")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Ошибка миграции: {e}")
        raise

if __name__ == "__main__":
    main()