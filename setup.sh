#!/bin/bash

echo "🚀 Настройка Immigration Bot"
echo "=============================="

# Проверка Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен. Установи Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✅ Docker найден"

# Проверка docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose не установлен"
    exit 1
fi

echo "✅ docker-compose найден"

# Копирование .env
if [ ! -f .env ]; then
    echo "📝 Создаю .env файл..."
    cp .env.example .env
    echo "⚠️  ВАЖНО: Отредактируй .env файл и добавь:"
    echo "   - TELEGRAM_BOT_TOKEN"
    echo "   - ADMIN_TELEGRAM_IDS"
    echo ""
    read -p "Нажми Enter когда отредактируешь .env..."
fi

# Запуск контейнеров
echo "🐳 Запускаю PostgreSQL и Ollama..."
docker-compose up -d

echo "⏳ Жду запуска сервисов (30 сек)..."
sleep 30

# Проверка PostgreSQL
echo "🔍 Проверяю PostgreSQL..."
until docker-compose exec -T postgres pg_isready -U bot_user -d immigration_bot &> /dev/null; do
    echo "   Жду PostgreSQL..."
    sleep 2
done
echo "✅ PostgreSQL готов"

# Скачивание моделей Ollama
echo "📥 Скачиваю модели Ollama..."
echo "   Это может занять 10-15 минут в зависимости от скорости интернета"

OLLAMA_CONTAINER=$(docker-compose ps -q ollama)

echo "   Загружаю llama3.1:8b (4.7GB)..."
docker exec -it $OLLAMA_CONTAINER ollama pull llama3.1:8b

echo "   Загружаю nomic-embed-text (274MB)..."
docker exec -it $OLLAMA_CONTAINER ollama pull nomic-embed-text

echo "✅ Модели загружены"

# Установка Python зависимостей
echo "📦 Устанавливаю Python зависимости..."
pip install -r requirements.txt

echo ""
echo "✅ Настройка завершена!"
echo ""
echo "🎯 Следующие шаги:"
echo "   1. Убедись что в .env указан TELEGRAM_BOT_TOKEN"
echo "   2. Запусти бота: python -m bot.main"
echo "   3. Напиши /start боту в Telegram"
echo ""
echo "📊 Полезные команды:"
echo "   docker-compose logs -f          # Логи сервисов"
echo "   docker-compose ps               # Статус контейнеров"
echo "   docker-compose down             # Остановить все"
echo "   docker-compose restart ollama   # Перезапустить Ollama"
echo ""