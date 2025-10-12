import os
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from database import init_db
from bot.handlers.user import start_command, help_command, handle_question
from bot.handlers.admin import answer_command, pending_command, stats_command

load_dotenv()


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    print(f"Update {update} caused error {context.error}")
    
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "😕 Произошла ошибка при обработке запроса. Попробуй позже."
        )


def main():
    """Запуск бота"""
    print("🔧 Инициализация базы данных...")
    init_db()
    
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN не установлен в .env")
    
    application = Application.builder().token(token).build()
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    
    application.add_handler(CommandHandler("answer", answer_command))
    application.add_handler(CommandHandler("pending", pending_command))
    application.add_handler(CommandHandler("stats", stats_command))
    
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question)
    )
    
    application.add_error_handler(error_handler)
    
    print("🚀 Бот запущен!")
    print(f"📊 LLM Provider: {os.getenv('LLM_PROVIDER', 'ollama')}")
    print(f"🎯 Confidence Threshold: {os.getenv('CONFIDENCE_THRESHOLD', '0.7')}")
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()