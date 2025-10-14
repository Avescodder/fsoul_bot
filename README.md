# 🤖 Telegram Бот - Консультант по Иммиграции в Португалию

Умный бот-ассистент с AI, который помогает отвечать на вопросы об иммиграции в Португалию. Использует RAG (Retrieval-Augmented Generation) для поиска релевантной информации в базе знаний.

## ✨ Возможности

- 🧠 **AI-ассистент** - автоматически отвечает на вопросы используя базу знаний
- 📚 **База знаний** - хранит проверенные Q&A пары
- 👨‍💼 **Эскалация** - передает сложные вопросы администраторам
- 🎯 **Обучение** - автоматически добавляет ответы админов в базу знаний
- 💬 **Естественное общение** - отвечает как живой консультант Сергей
- 📊 **Статистика** - отслеживает эффективность работы

## 🏗️ Архитектура

```
┌─────────────┐
│  Пользователь│
└──────┬──────┘
       │ вопрос
       ▼
┌──────────────────┐
│   Telegram Bot   │
└────────┬─────────┘
         │
    ┌────▼────┐
    │   RAG   │ ← Поиск похожих вопросов
    └────┬────┘
         │
    ┌────▼────┐
    │   LLM   │ ← OpenAI/Groq/Claude
    └────┬────┘
         │
    ┌────▼────┐
    │confidence│
    └────┬────┘
         │
    ┌────┴────┐
    │         │
   >0.7      <0.7
    │         │
    ▼         ▼
 Ответ    Админам
```

## 🚀 Быстрый старт

### Локальная разработка

```bash
# Клонируй репозиторий
git clone https://github.com/yourusername/immigration-bot.git
cd immigration-bot

# Создай виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установи зависимости
pip install -r requirements.txt

# Создай .env файл
cp .env.example .env
# Заполни своими данными

# Запусти бота
python -m bot.main
```

### Docker (рекомендуется)

```bash
# Создай .env файл
cp .env.example .env
# Заполни своими данными

# Запусти через Docker Compose
docker-compose up -d

# Проверь логи
docker-compose logs -f bot
```

## 📋 Требования

- Python 3.11+
- PostgreSQL 16+ с pgvector
- Docker & Docker Compose (для продакшена)
- API ключ одного из LLM провайдеров:
  - OpenAI (рекомендуется) - $0.35/1000 вопросов
  - Groq (бесплатно с лимитами)
  - Claude
  - Ollama (локально)

## ⚙️ Конфигурация

### Переменные окружения (.env)

```bash
# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
ADMIN_TELEGRAM_IDS=123456789,987654321

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/immigration_bot

# LLM Provider (выбери один)
LLM_PROVIDER=openai  # или groq, claude, ollama

# OpenAI (если LLM_PROVIDER=openai)
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Настройки
CONFIDENCE_THRESHOLD=0.7  # Порог для эскалации админу
```

Полный список переменных см. в [.env.example](.env.example)

## 📚 Команды бота

### Для пользователей:
- `/start` - Начать работу с ботом
- `/help` - Справка
- Просто напиши вопрос - бот ответит автоматически

### Для администраторов:
- `/pending` - Показать вопросы в ожидании
- `/stats` - Статистика работы бота
- `/answer <ID> <текст>` - Ответить на вопрос
- `/teach` - Добавить Q&A в базу знаний вручную

## 📊 Мониторинг

### Логи
```bash
# Реал-тайм логи
docker-compose logs -f bot

# Последние 100 строк
docker-compose logs --tail=100 bot
```

### Статистика в боте
```
/stats
```
Покажет:
- Количество пользователей
- Обработано вопросов
- % автоматических ответов
- Размер базы знаний

### Использование ресурсов
```bash
docker stats
```

## 🗄️ База данных

### Бэкап
```bash
docker-compose exec db pg_dump -U botuser immigration_bot > backup.sql
```

### Восстановление
```bash
make restore FILE=backup.sql
```

### Подключение к БД
```bash
make db-shell
# или
docker-compose exec db psql -U botuser -d immigration_bot
```

## 🛠️ Разработка

### Структура проекта
```
immigration-bot/
├── bot/
│   ├── handlers/          # Обработчики команд
│   │   ├── user.py       # Команды пользователей
│   │   └── admin.py      # Команды админов
│   ├── llm/              # LLM провайдеры
│   │   ├── openai.py
│   │   ├── groq.py
│   │   ├── claude.py
│   │   └── ollama.py
│   └── main.py           # Точка входа
├── database/
│   ├── models.py         # SQLAlchemy модели
│   └── __init__.py
├── utils/
│   └── rag.py           # RAG система
├── scripts/
│   └── import_knowledge.py  # Импорт знаний
├── .github/
│   └── workflows/       # CI/CD пайплайны
├── docker-compose.yml
├── Dockerfile
├── Makefile            # Упрощенные команды
└── requirements.txt
```

### Workflow
```bash
# Создай ветку для фичи
git checkout -b feature/new-feature

# Разработка...
git add .
git commit -m "feat: add new feature"

# Пушь в dev для тестирования
git checkout dev
git merge feature/new-feature
git push origin dev

# GitHub Actions создаст PR в main
# После мерджа - автоматический деплой
```

## 🐛 Troubleshooting

### Бот не запускается
```bash
docker-compose logs bot

cat .env

docker-compose down
docker-compose up -d --build
```

### Не подключается к БД
```bash
docker-compose ps

docker-compose exec db pg_isready -U botuser
```

## 🤝 Вклад в проект

1. Fork репозиторий
2. Создай ветку (`git checkout -b feature/amazing`)
3. Коммит изменений (`git commit -m 'feat: add amazing feature'`)
4. Push в ветку (`git push origin feature/amazing`)
5. Открой Pull Request

## 👥 Авторы

- Avescodder 

## 🙏 Благодарности

- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- [OpenAI](https://openai.com)
- [pgvector](https://github.com/pgvector/pgvector)

## 📞 Поддержка

- 🐛 Нашел баг? Создай [Issue](https://github.com/yourusername/immigration-bot/issues)
- 💡 Есть идея? Открой [Discussion](https://github.com/yourusername/immigration-bot/discussions)
- 📧 Email: morozovvsevolod24@gmail.com
- 💬 Telegram: [@avescodder]

---

⭐ Если проект полезен, поставь звездочку на GitHub!
