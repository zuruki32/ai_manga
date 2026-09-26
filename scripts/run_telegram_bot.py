#!/usr/bin/env python3
"""Run Irisekai Telegram translation bot.

  export TELEGRAM_BOT_TOKEN=...   # or put in .env
  python scripts/run_telegram_bot.py
"""

from manga_ai.telegram.bot import run_bot

if __name__ == "__main__":
    run_bot()
