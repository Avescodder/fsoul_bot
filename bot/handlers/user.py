from telegram import Update
from telegram.ext import ContextTypes
from database import get_db
from database.models import User, Question, PendingQuestion
from utils.improved_rag import ImprovedRAGSystemWithTavily
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
    
    if is_admin(user_id):
        await update.message.reply_text(
            "👋 Привет, администратор!\n\n"
            "Доступные команды:\n"
            "/pending - вопросы в ожидании\n"
            "/stats - статистика бота\n"
            "/answer <ID> <текст> - ответить на вопрос"
        )
    else:
        await update.message.reply_text(
            "Здравствуйте! Меня зовут Сергей.\n\n"
            "Я эксперт по иммиграции в Португалию с более чем 10-летним опытом. "
            "Помогу вам с визами, ВНЖ, налогообложением и другими вопросами переезда.\n\n"
            "Задавайте ваши вопросы на русском, английском или португальском языке!"
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    user_id = update.effective_user.id
    
    if is_admin(user_id):
        await update.message.reply_text(
            "ℹ️ Справка для администратора:\n\n"
            "👨‍💼 Команды:\n"
            "/pending - очередь вопросов\n"
            "/stats - статистика\n"
            "/answer <ID> <текст> - ответить\n\n"
            "🤖 Бот автоматически учится на ваших ответах!"
        )
    else:
        await update.message.reply_text(
            "Я помогу вам с:\n\n"
            "✈️ Визами (D7, Golden Visa, D2)\n"
            "🏠 ВНЖ и гражданством\n"
            "💼 Открытием бизнеса\n"
            "💰 Налогообложением и NHR\n"
            "📄 Документами и процедурами\n\n"
            "Просто задайте вопрос на любом языке!"
        )


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Улучшенный обработчик вопросов с контекстом"""
    question_text = update.message.text
    user_tg_id = update.effective_user.id
    
    lang = detect_language(question_text)
    
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
        
        rag = ImprovedRAGSystemWithTavily(
        llm=llm,
        tavily_api_key=os.getenv("TAVILY_API_KEY")
        )
        answer, confidence, context_data = await rag.get_answer_with_web_search(
            db=db,
            question=question_text,
            user_id=user.id,
            use_web_search=False,  
            search_depth="basic"
        )
                
        threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
        
        should_escalate = should_escalate_to_admin(
            question_text=question_text,
            confidence=confidence,
            threshold=threshold,
            context_available=len(context_data) > 0
        )
        
        print(f"📊 Q: {question_text[:50]}... | Conf: {confidence:.2%} | Escalate: {should_escalate}")
        
        if not should_escalate:
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
            
            escalation_messages = {
                'ru': "Ваш вопрос требует детального изучения. Я проконсультируюсь с коллегами и вернусь с точным ответом в ближайшее время.",
                'en': "Your question requires detailed analysis. I'll consult with colleagues and get back to you with a precise answer shortly.",
                'pt': "Sua pergunta requer análise detalhada. Vou consultar colegas e retornarei com uma resposta precisa em breve."
            }
            
            await update.message.reply_text(escalation_messages.get(lang, escalation_messages['ru']))
            
            await notify_admins(update, context, question.id, user, question_text, confidence)


def should_escalate_to_admin(
    question_text: str,
    confidence: float,
    threshold: float,
    context_available: bool
) -> bool:
    """
    Умная логика эскалации к админу
    """
    
    simple_keywords = [
        'привет', 'спасибо', 'здравствуй', 'пока', 'благодарю',
        'hi', 'hello', 'thanks', 'thank you', 'bye',
        'olá', 'obrigado', 'tchau'
    ]
    
    q_lower = question_text.lower()
    if any(kw in q_lower for kw in simple_keywords) and len(question_text.split()) < 10:
        return False
    
    if context_available and confidence >= 0.65:
        return False
    
    if confidence >= threshold:
        return False
    
    critical_keywords = [
        'депортация', 'отказ', 'судебный', 'апелляция',
        'deportation', 'refusal', 'court', 'appeal',
        'deportação', 'recusa', 'tribunal'
    ]
    
    if any(kw in q_lower for kw in critical_keywords) and confidence < 0.75:
        return True
    
    return confidence < threshold


def detect_language(text: str) -> str:
    """Простое определение языка вопроса"""
    text_lower = text.lower()
    
    if any(c in 'абвгдежзийклмнопрстуфхцчшщъыьэюя' for c in text_lower):
        return 'ru'
    
    pt_words = ['você', 'não', 'sim', 'obrigado', 'por favor', 'está', 'também', 'quando']
    if any(word in text_lower for word in pt_words):
        return 'pt'
    
    return 'en'


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
        f"💬 Ответить: /answer {question_id} <текст>"
    )
    
    bot = context.bot if hasattr(context, 'bot') else update.get_bot()
    
    for admin_id in admin_ids:
        try:
            await bot.send_message(chat_id=admin_id, text=message_text)
        except Exception as e:
            print(f"⚠️ Не удалось отправить админу {admin_id}: {e}")