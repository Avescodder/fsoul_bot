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
    """Обработчик команды /start - теперь просто приветствие"""
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
    
    if is_admin(user_id):
        await update.message.reply_text(
            "👋 Привет! Ты администратор.\n\n"
            "Доступные команды:\n"
            "/pending - вопросы в ожидании\n"
            "/stats - статистика бота\n"
            "/answer <ID> <текст> - ответить на вопрос"
        )
    else:
        await update.message.reply_text(
            "Здравствуйте! Меня зовут Сергей.\n\n"
            "Я помогаю с вопросами по иммиграции в Португалию. "
            "Задавайте свои вопросы, и я постараюсь помочь вам максимально подробно."
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    user_id = update.effective_user.id
    
    if is_admin(user_id):
        await update.message.reply_text(
            "ℹ️ Справка для админа:\n\n"
            "👨‍💼 Команды администратора:\n"
            "/pending - показать вопросы в очереди\n"
            "/stats - статистика работы бота\n"
            "/answer <ID> <текст> - ответить на вопрос\n\n"
            "💡 Как отвечать на вопросы:\n"
            "1. Получаешь уведомление о новом вопросе\n"
            "2. Используешь /answer <ID> <твой ответ>\n"
            "3. Ответ автоматически отправится пользователю"
        )
    else:
        await update.message.reply_text(
            "Меня зовут Сергей, я специалист по иммиграции в Португалию.\n\n"
            "Просто напишите мне свой вопрос, и я постараюсь помочь вам с максимальной детализацией."
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
        answer, confidence, context_data = await rag.get_answer(db, question_text)
        
        threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
        
        print(f"📊 Вопрос: {question_text[:50]}... | Уверенность: {confidence:.2%}")
        
        if confidence >= threshold:
            question.answer_text = answer
            question.confidence_score = confidence
            question.answered_by_ai = True
            question.status = "answered"
            question.answered_at = datetime.utcnow()
            db.commit()
            
            await update.message.reply_text(answer)
            
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
                "К сожалению, я не уверен в ответе на ваш вопрос. "
                "Позвольте мне проконсультироваться с коллегами, "
                "и я обязательно вернусь к вам с точным ответом в ближайшее время."
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
        f"👤 От: {user.first_name or ''} {user.last_name or ''} "
        f"(@{user.username or 'без username'})\n"
        f"🆔 Telegram ID: {user.telegram_id}\n\n"
        f"📝 Вопрос:\n{question_text}\n\n"
        f"🤖 Уверенность AI: {confidence:.1%}\n\n"
        f"💬 Используй: /answer {question_id} <твой ответ>"
    )
    
    bot = context.bot if hasattr(context, 'bot') else update.get_bot()
    
    for admin_id in admin_ids:
        try:
            await bot.send_message(chat_id=admin_id, text=message_text)
        except Exception as e:
            print(f"⚠️ Не удалось отправить админу {admin_id}: {e}")