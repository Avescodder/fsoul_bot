from telegram import Update
from telegram.ext import ContextTypes
from database import get_db
from database.models import User, Question, PendingQuestion
from utils.rag import RAGSystem
from bot.llm import get_llm
from bot.handlers.admin import is_admin
import os
from datetime import datetime


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user_id = update.effective_user.id
    
    with get_db() as db:
        user = db.query(User).filter(User.telegram_id == user_id).first()
        
        if not user:
            user = User(
                telegram_id=user_id,
                username=update.effective_user.username,
                first_name=update.effective_user.first_name,
                last_name=update.effective_user.last_name
            )
            db.add(user)
            db.commit()
    
    if is_admin():
        await update.message.reply_text(
            "👋 Привет, админ! Ты можешь использовать специальные команды:\n\n"
            "👨‍💼 Команды админа:\n"
            "/pending - вопросы в ожидании\n"
            "/stats - статистика бота\n"
            "/answer <ID> <текст> - ответить на вопрос\n\n"
            "Также ты можешь задавать вопросы как обычный пользователь."
        )
    else:
        await update.message.reply_text(
            "👋 Привет! Я бот-помощник по вопросам иммиграции в Португалию.\n\n"
            "Задай мне любой вопрос, и я постараюсь помочь. "
            "Если я не смогу ответить сразу, передам вопрос специалисту."
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    user_id = update.effective_user.id
    
    admin_ids = os.getenv("ADMIN_TELEGRAM_IDS", "").split(",")
    admin_ids = [int(aid.strip()) for aid in admin_ids if aid.strip()]
    is_admin = user_id in admin_ids
    
    if is_admin:
        await update.message.reply_text(
            "ℹ️ Справка для админа:\n\n"
            "👨‍💼 Команды администратора:\n"
            "/pending - показать вопросы в очереди\n"
            "/stats - статистика работы бота\n"
            "/answer <ID> <текст> - ответить на вопрос\n\n"
            "👤 Команды пользователя:\n"
            "/start - начать работу\n"
            "/help - эта справка\n\n"
            "💡 Как отвечать на вопросы:\n"
            "1. Получаешь уведомление о новом вопросе\n"
            "2. Используешь /answer <ID> <твой ответ>\n"
            "3. Ответ автоматически отправится пользователю\n"
            "4. Бот сохранит Q&A в базу знаний и обучится"
        )
    else:
        await update.message.reply_text(
            "ℹ️ Как пользоваться ботом:\n\n"
            "• Просто напиши свой вопрос о иммиграции в Португалию\n"
            "• Я постараюсь ответить на основе своих знаний\n"
            "• Если вопрос сложный, я передам его специалисту\n"
            "• Ты получишь ответ от эксперта\n\n"
            "Команды:\n"
            "/start - начать работу\n"
            "/help - эта справка"
        )


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик вопросов пользователей"""
    question_text = update.message.text
    user_tg_id = update.effective_user.id
    
    await update.message.chat.send_action("typing")
    
    with get_db() as db:
        user = db.query(User).filter(User.telegram_id == user_tg_id).first()
        if not user:
            user = User(
                telegram_id=user_tg_id,
                username=update.effective_user.username,
                first_name=update.effective_user.first_name,
                last_name=update.effective_user.last_name
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        
        llm = get_llm()
        question_embedding = await llm.generate_embedding(question_text)
        
        question = Question(
            user_id=user.id,
            message_id=update.message.message_id,
            question_text=question_text,
            question_embedding=question_embedding,
            status="processing"
        )
        db.add(question)
        db.commit()
        db.refresh(question)
        
        rag = RAGSystem(llm)
        answer, confidence, context = await rag.get_answer(db, question_text)
        
        threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
        
        if confidence >= threshold:
            question.answer_text = answer
            question.confidence_score = confidence
            question.answered_by_ai = True
            question.status = "answered"
            question.answered_at = datetime.utcnow()
            db.commit()
            
            await update.message.reply_text(
                f"{answer}\n\n"
                f"🤖 _Автоматический ответ (уверенность: {confidence:.0%})_",
                parse_mode="Markdown"
            )
            
        else:
            question.confidence_score = confidence
            question.status = "escalated"
            db.commit()
            
            pending = PendingQuestion(
                question_id=question.id,
                user_telegram_id=user_tg_id
            )
            db.add(pending)
            db.commit()
            
            await update.message.reply_text(
                "🤔 Сложный вопрос! Я передал его нашим специалистам.\n"
                "Ожидай ответ в течение нескольких часов."
            )
            
            await notify_admins(update, context, question.id, user, question_text, confidence)


async def notify_admins(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    question_id: int,
    user: User,
    question_text: str,
    confidence: float
):
    """Отправляет вопрос админам"""
    admin_ids = os.getenv("ADMIN_TELEGRAM_IDS", "").split(",")
    admin_ids = [int(aid.strip()) for aid in admin_ids if aid.strip()]
    
    message_text = (
        f"❓ Новый вопрос #{question_id}\n\n"
        f"От: {user.first_name or ''} {user.last_name or ''} "
        f"(@{user.username or 'без username'})\n"
        f"ID: {user.telegram_id}\n\n"
        f"Вопрос:\n{question_text}\n\n"
        f"🤖 Уверенность AI: {confidence:.0%}\n\n"
        f"Используй /answer {question_id} <твой ответ> для ответа"
    )
    
    bot = context.bot if context else update.get_bot()
    
    for admin_id in admin_ids:
        try:
            await bot.send_message(chat_id=admin_id, text=message_text)
        except Exception as e:
            print(f"Не удалось отправить админу {admin_id}: {e}")