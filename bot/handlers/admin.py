from telegram import Update
from telegram.ext import ContextTypes
from database import get_db
from database.models import Question, User, PendingQuestion
from utils.improved_rag import ImprovedRAGSystemWithTavily
from bot.llm import get_llm
import os
from datetime import datetime


def is_admin(user_id: int) -> bool:
    """Проверяет, является ли пользователь админом"""
    admin_ids = os.getenv("ADMIN_TELEGRAM_IDS", "").split(",")
    admin_ids = [int(aid.strip()) for aid in admin_ids if aid.strip()]
    return user_id in admin_ids


async def answer_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработчик команды /answer для админов
    Формат: /answer <question_id> <ответ>
    """
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Эта команда доступна только администраторам")
        return
    
    if len(context.args) < 2:
        await update.message.reply_text(
            "❌ Неправильный формат.\n\n"
            "Используй: /answer <ID вопроса> <твой ответ>\n\n"
            "Пример: /answer 42 Для получения визы нужно подать документы..."
        )
        return
    
    try:
        question_id = int(context.args[0])
        answer_text = " ".join(context.args[1:])
    except ValueError:
        await update.message.reply_text("❌ ID вопроса должен быть числом")
        return
    
    with get_db() as db:
        question = db.query(Question).filter(Question.id == question_id).first()
        
        if not question:
            await update.message.reply_text(f"❌ Вопрос #{question_id} не найден")
            return
        
        if question.status == "answered":
            await update.message.reply_text(f"⚠️ На вопрос #{question_id} уже ответили")
            return
        
        question.answer_text = answer_text
        question.answered_by_ai = False
        question.answered_by_admin_id = update.effective_user.id
        question.status = "answered"
        question.answered_at = datetime.utcnow()
        db.commit()
        
        llm = get_llm()
        rag = ImprovedRAGSystemWithTavily(
            llm=llm,
            tavily_api_key=os.getenv("TAVILY_API_KEY")
        )
        await rag.add_to_knowledge_base(
            db,
            question=question.question_text,
            answer=answer_text,
            source="admin",
            verified=True
        )
        
        db.query(PendingQuestion).filter(
            PendingQuestion.question_id == question_id
        ).delete()
        db.commit()
        
        try:
            bot = context.bot if hasattr(context, 'bot') else update.get_bot()
            await bot.send_message(
                chat_id=question.user.telegram_id,
                text=answer_text  
            )
            
            await update.message.reply_text(
                f"✅ Ответ отправлен пользователю!\n"
                f"📝 Вопрос #{question_id} закрыт\n"
                f"📚 Добавлено в базу знаний"
            )
        except Exception as e:
            await update.message.reply_text(
                f"⚠️ Ответ сохранен в базу, но не удалось отправить пользователю:\n{e}\n\n"
                f"📚 Q&A добавлен в базу знаний"
            )


async def pending_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает список вопросов в ожидании для админов"""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Эта команда доступна только администраторам")
        return
    
    with get_db() as db:
        pending_questions = db.query(PendingQuestion, Question, User).join(
            Question, PendingQuestion.question_id == Question.id
        ).join(
            User, Question.user_id == User.id
        ).filter(
            Question.status == "escalated"
        ).order_by(Question.created_at.desc()).all()
        
        if not pending_questions:
            await update.message.reply_text("✅ Нет вопросов в ожидании!")
            return
        
        message = "📋 *Вопросы в ожидании ответа:*\n\n"
        
        for pending, question, user in pending_questions:
            question_preview = question.question_text
            if len(question_preview) > 100:
                question_preview = question_preview[:97] + "..."
            
            time_str = question.created_at.strftime('%d.%m %H:%M')
            confidence_pct = f"{question.confidence_score:.0%}" if question.confidence_score else "N/A"
            
            message += (
                f"*#{question.id}* | {user.first_name or 'User'} (@{user.username or 'нет'})\n"
                f"📝 _{question_preview}_\n"
                f"🤖 AI уверенность: {confidence_pct}\n"
                f"⏰ {time_str}\n\n"
            )
        
        message += "💬 Используй: `/answer <ID> <ответ>`"
        
        await update.message.reply_text(message, parse_mode="Markdown")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает статистику для админов"""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Эта команда доступна только администраторам")
        return
    
    with get_db() as db:
        total_questions = db.query(Question).count()
        answered_by_ai = db.query(Question).filter(
            Question.answered_by_ai
        ).count()
        answered_by_admin = db.query(Question).filter(
            not Question.answered_by_ai,
            Question.status == "answered"
        ).count()
        pending = db.query(Question).filter(
            Question.status == "escalated"
        ).count()
        total_users = db.query(User).count()
        
        from database.models import KnowledgeBase
        kb_size = db.query(KnowledgeBase).count()
        
        avg_response_time = None
        answered_questions = db.query(Question).filter(
            Question.answered_at.isnot(None),
            Question.created_at.isnot(None)
        ).all()
        
        if answered_questions:
            total_time = sum([
                (q.answered_at - q.created_at).total_seconds() 
                for q in answered_questions
            ])
            avg_response_time = total_time / len(answered_questions) / 3600 
        
        message = (
            "📊 *Статистика бота:*\n\n"
            f"👥 Пользователей: `{total_users}`\n"
            f"❓ Всего вопросов: `{total_questions}`\n"
            f"🤖 Ответов от AI: `{answered_by_ai}`\n"
            f"👨‍💼 Ответов от админов: `{answered_by_admin}`\n"
            f"⏳ В ожидании: `{pending}`\n"
            f"📚 База знаний: `{kb_size}` записей\n"
        )
        
        if avg_response_time:
            message += f"⏱ Среднее время ответа: `{avg_response_time:.1f}` ч\n"
        
        if total_questions > 0:
            ai_percentage = (answered_by_ai / total_questions) * 100
            message += f"\n💡 AI решает *{ai_percentage:.1f}%* вопросов автоматически"
        
        await update.message.reply_text(message, parse_mode="Markdown")