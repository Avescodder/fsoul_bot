# Portugal Immigration Bot

A Telegram bot that answers immigration questions using RAG + LLM. When confidence is low, it escalates to admins — and learns from their replies.

## Stack

- **Python 3.11+** with python-telegram-bot
- **PostgreSQL + pgvector** for vector search
- **LLM**: OpenAI (default), Groq, Claude, or Ollama

## Quickstart

```bash
cp .env.example .env        # fill in your credentials
docker-compose up -d        # start bot + db
docker-compose logs -f bot  # check logs
```

## Configuration

Key variables in `.env`:

```env
TELEGRAM_BOT_TOKEN=...
ADMIN_TELEGRAM_IDS=123456789,987654321
DATABASE_URL=postgresql://user:pass@localhost:5432/immigration_bot

LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

CONFIDENCE_THRESHOLD=0.7   # below this → escalate to admin
```

See `.env.example` for the full list.

## Bot Commands

| Command | Who | Description |
|---|---|---|
| `/start` | Users | Start the bot |
| `/help` | Users | Help message |
| `/pending` | Admins | View unanswered questions |
| `/answer <ID> <text>` | Admins | Reply to a question |
| `/teach` | Admins | Manually add Q&A to knowledge base |
| `/stats` | Admins | Usage statistics |

## Project Structure

```
bot/
  handlers/     # user.py, admin.py
  llm/          # openai.py, groq.py, claude.py, ollama.py
  main.py
database/
  models.py
utils/
  rag.py
scripts/
  import_knowledge.py
```

## DB Management

```bash
# Backup
docker-compose exec db pg_dump -U botuser immigration_bot > backup.sql

# Shell
docker-compose exec db psql -U botuser -d immigration_bot
```

## Contact

- Email: morozovvsevolod24@gmail.com
- Telegram: [@avescodder](https://t.me/avescodder)
- Bugs / ideas: [open an issue](https://github.com/yourusername/immigration-bot/issues)
