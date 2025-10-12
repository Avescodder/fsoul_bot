.PHONY: help setup start stop restart logs db-init db-seed db-stats test-llm clean

help: ## Показать справку
	@echo "📚 Доступные команды:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Первичная настройка (Docker + модели)
	@chmod +x setup.sh
	@./setup.sh

start: ## Запустить бота
	python -m bot.main

docker-up: ## Запустить Docker контейнеры
	docker-compose up -d
	@echo "⏳ Жду запуска сервисов..."
	@sleep 10
	@echo "✅ Сервисы запущены"

docker-down: ## Остановить Docker контейнеры
	docker-compose down

docker-restart: ## Перезапустить Docker контейнеры
	docker-compose restart

logs: ## Показать логи Docker
	docker-compose logs -f

db-init: ## Инициализировать базу данных
	python -c "from database import init_db; init_db()"

db-seed: ## Заполнить базу знаний начальными данными
	python -c "from utils.db_manager import seed_knowledge_base; import asyncio; asyncio.run(seed_knowledge_base())"

db-stats: ## Показать статистику БД
	python -c "from utils.db_manager import show_stats; show_stats()"

db-manager: ## Запустить менеджер БД
	python utils/db_manager.py

test-llm: ## Тест LLM провайдера
	python test_llm.py

install: ## Установить зависимости
	pip install -r requirements.txt

clean: ## Очистить временные файлы
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete

pull-models: ## Скачать модели Ollama
	docker exec $$(docker-compose ps -q ollama) ollama pull llama3.1:8b
	docker exec $$(docker-compose ps -q ollama) ollama pull nomic-embed-text

ollama-list: ## Показать установленные модели
	docker exec $$(docker-compose ps -q ollama) ollama list

dev: docker-up install db-init db-seed ## Полная настройка для разработки
	@echo "✅ Разработческое окружение готово!"
	@echo "🚀 Запусти бота: make start"