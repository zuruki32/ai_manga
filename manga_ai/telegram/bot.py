"""Telegram bot for Irisekai Translation — projects, names, ZIP chapters.

Security: set TELEGRAM_BOT_TOKEN in environment / .env — never commit the token.
If a token was shared in chat, revoke it in @BotFather and create a new one.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

from manga_ai.logging import get_logger, setup_logging
from manga_ai.telegram.jobs import retranslate_with_new_names, run_chapter_from_zip
from manga_ai.telegram.projects import ProjectStore
from manga_ai.telegram.prompt_review import review_translation_prompt

logger = get_logger("manga_ai.telegram.bot")

# In-memory per-user conversation state (fine for single-operator bot)
_STATE: Dict[int, Dict] = {}


def _kb(buttons):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    rows = []
    for row in buttons:
        rows.append(
            [InlineKeyboardButton(text=t, callback_data=d) for t, d in row]
        )
    return InlineKeyboardMarkup(rows)


def main_menu():
    return _kb(
        [
            [("📁 New manhwa project", "proj_new")],
            [("📚 My projects", "proj_list")],
            [("ℹ️ Help", "help")],
        ]
    )


def project_menu(slug: str):
    return _kb(
        [
            [("🔤 Add character name (EN+FA)", f"name_add:{slug}")],
            [("📋 Show names", f"name_list:{slug}")],
            [("📦 Upload chapter ZIP", f"zip:{slug}")],
            [("✍️ Add translation style tip", f"prompt_add:{slug}")],
            [("✅ Approve pending style tip", f"prompt_ok:{slug}")],
            [("🏠 Main menu", "home")],
        ]
    )


async def cmd_start(update, context):
    uid = update.effective_user.id
    _STATE[uid] = {"mode": None}
    text = (
        "سلام! به *Irisekai Translation Bot* خوش اومدی.\n\n"
        "۱) پروژه مانها بساز\n"
        "۲) اسم شخصیت‌ها رو EN + فارسی اضافه کن\n"
        "۳) ZIP صفحات فصل رو بفرست\n"
        "۴) بعد ترجمه، اسم‌های جاافتاده رو می‌گم تا اضافه کنی و دوباره ترجمه بشه\n\n"
        "نکته: دستور لحن/ترجمه فقط بعد از بررسی AI به کانفیگ اضافه می‌شه. "
        "اسم‌ها مستقیم ذخیره می‌شن."
    )
    await update.message.reply_text(text, reply_markup=main_menu(), parse_mode="Markdown")


async def on_callback(update, context):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    data = q.data or ""
    store: ProjectStore = context.bot_data["store"]

    if data == "home":
        _STATE[uid] = {"mode": None}
        await q.edit_message_text("Main menu:", reply_markup=main_menu())
        return

    if data == "help":
        await q.edit_message_text(
            "Flow:\n"
            "• New project → folder + config.yaml per manhwa\n"
            "• Add names as: `English | فارسی`\n"
            "• Send a ZIP of page images\n"
            "• Bot reports missing names → add → retranslate\n"
            "• Style tips need AI review then Approve\n",
            reply_markup=main_menu(),
            parse_mode="Markdown",
        )
        return

    if data == "proj_new":
        _STATE[uid] = {"mode": "await_project_title"}
        await q.edit_message_text("اسم مانها / پروژه رو بفرست (مثلاً Suddenly Became A Princess):")
        return

    if data == "proj_list":
        projects = store.list_projects()
        if not projects:
            await q.edit_message_text("هنوز پروژه‌ای نداری.", reply_markup=main_menu())
            return
        buttons = [[(p.get("title") or p["slug"], f"proj_open:{p['slug']}")] for p in projects]
        buttons.append([("🏠 Main menu", "home")])
        await q.edit_message_text("پروژه‌ها:", reply_markup=_kb(buttons))
        return

    if data.startswith("proj_open:"):
        slug = data.split(":", 1)[1]
        meta = store.get(slug)
        if not meta:
            await q.edit_message_text("Project not found.", reply_markup=main_menu())
            return
        gloss = store.load_glossary(slug)
        await q.edit_message_text(
            f"Project: *{meta.get('title')}*\n"
            f"slug: `{slug}`\n"
            f"names: {len(gloss)}\n"
            f"pending style tips: {len(meta.get('pending_prompts') or [])}",
            reply_markup=project_menu(slug),
            parse_mode="Markdown",
        )
        return

    if data.startswith("name_add:"):
        slug = data.split(":", 1)[1]
        _STATE[uid] = {"mode": "await_name", "slug": slug}
        await q.edit_message_text(
            "اسم رو این شکلی بفرست:\n`EnglishName | فارسی`\n"
            "مثال: `Jennette | جنت`",
            parse_mode="Markdown",
        )
        return

    if data.startswith("name_list:"):
        slug = data.split(":", 1)[1]
        gloss = store.load_glossary(slug)
        if not gloss:
            await q.edit_message_text("هنوز اسمی ثبت نشده.", reply_markup=project_menu(slug))
            return
        # de-dupe display by lowercase
        seen = set()
        lines = []
        for en, fa in gloss.items():
            key = en.lower()
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"• {en} → {fa}")
        await q.edit_message_text(
            "Names:\n" + "\n".join(lines[:80]),
            reply_markup=project_menu(slug),
        )
        return

    if data.startswith("zip:"):
        slug = data.split(":", 1)[1]
        _STATE[uid] = {"mode": "await_zip", "slug": slug}
        await q.edit_message_text(
            "ZIP صفحات فصل رو بفرست (png/jpg داخلش).\n"
            "اختیاری: کپشن اسم فصل رو هم بنویس."
        )
        return

    if data.startswith("prompt_add:"):
        slug = data.split(":", 1)[1]
        _STATE[uid] = {"mode": "await_prompt", "slug": slug}
        await q.edit_message_text(
            "دستور لحن/ترجمه رو بفرست (فارسی یا انگلیسی).\n"
            "اول AI بررسی می‌کنه؛ اگر OK بود با دکمه Approve به کانفیگ اضافه می‌شه."
        )
        return

    if data.startswith("prompt_ok:"):
        slug = data.split(":", 1)[1]
        try:
            text = store.approve_pending_prompt(slug)
            await q.edit_message_text(
                f"Style tip added to config:\n{text}",
                reply_markup=project_menu(slug),
            )
        except Exception as e:
            await q.edit_message_text(str(e), reply_markup=project_menu(slug))
        return

    if data.startswith("missing_add:"):
        # missing_add:slug:Name
        parts = data.split(":", 2)
        if len(parts) < 3:
            return
        _, slug, en_name = parts
        ch = context.user_data.get("last_chapter_dir") or (_STATE.get(uid) or {}).get("chapter_dir")
        _STATE[uid] = {
            "mode": "await_missing_fa",
            "slug": slug,
            "en_name": en_name,
            "chapter_dir": ch,
        }
        await q.edit_message_text(
            f"فارسیِ `{en_name}` رو بفرست:",
            parse_mode="Markdown",
        )
        return

    if data.startswith("retranslate:"):
        slug = data.split(":", 1)[1]
        ch = context.user_data.get("last_chapter_dir")
        if not ch:
            await q.edit_message_text("No chapter to retranslate.", reply_markup=project_menu(slug))
            return
        await q.edit_message_text("Retranslating with updated names…")
        try:
            result = await asyncio.to_thread(
                retranslate_with_new_names, store, slug, Path(ch), {}
            )
            missing = result.get("missing_names") or []
            msg = "Done retranslate.\n"
            if missing:
                msg += "Still missing:\n" + "\n".join(f"• {n}" for n in missing[:30])
            else:
                msg += "No obvious missing names left."
            path = result.get("scanlation_fa")
            await q.message.reply_text(msg, reply_markup=project_menu(slug))
            if path and Path(path).exists():
                with open(path, "rb") as fh:
                    await q.message.reply_document(document=fh, filename="chapter_scanlation_fa.txt")
        except Exception as e:
            logger.exception("retranslate failed")
            await q.message.reply_text(f"Failed: {e}", reply_markup=project_menu(slug))
        return


async def on_text(update, context):
    uid = update.effective_user.id
    st = _STATE.get(uid) or {}
    mode = st.get("mode")
    store: ProjectStore = context.bot_data["store"]
    text = (update.message.text or "").strip()

    if mode == "await_project_title":
        meta = store.create(text)
        _STATE[uid] = {"mode": None, "slug": meta["slug"]}
        await update.message.reply_text(
            f"Project created: *{meta['title']}*\n`{meta['slug']}`\n"
            f"Folder: `{store.path(meta['slug'])}`",
            reply_markup=project_menu(meta["slug"]),
            parse_mode="Markdown",
        )
        return

    if mode == "await_name":
        slug = st["slug"]
        if "|" not in text:
            await update.message.reply_text("فرمت: `English | فارسی`", parse_mode="Markdown")
            return
        en, fa = [x.strip() for x in text.split("|", 1)]
        store.add_name(slug, en, fa)
        _STATE[uid]["mode"] = None
        await update.message.reply_text(
            f"Saved: {en} → {fa}",
            reply_markup=project_menu(slug),
        )
        return

    if mode == "await_missing_fa":
        slug = st["slug"]
        en = st.get("en_name") or ""
        fa = text
        store.add_name(slug, en, fa)
        ch = st.get("chapter_dir") or context.user_data.get("last_chapter_dir")
        _STATE[uid]["mode"] = None
        buttons = [
            [("🔁 Retranslate chapter now", f"retranslate:{slug}")],
            [("➕ Add another name", f"name_add:{slug}")],
            [("📁 Project menu", f"proj_open:{slug}")],
        ]
        await update.message.reply_text(
            f"Saved {en} → {fa}. Retranslate when ready.",
            reply_markup=_kb(buttons),
        )
        return

    if mode == "await_prompt":
        slug = st["slug"]
        review = review_translation_prompt(text)
        store.add_pending_prompt(slug, text, review)
        _STATE[uid]["mode"] = None
        if review["approved"]:
            msg = (
                f"AI review: ✅ {review['summary']}\n\n"
                "اگر اوکیه، Approve رو بزن تا بره تو config."
            )
            await update.message.reply_text(msg, reply_markup=project_menu(slug))
        else:
            await update.message.reply_text(
                f"AI review: ❌ {review['summary']}\n"
                "اصلاحش کن و دوباره بفرست. به کانفیگ اضافه نشد.",
                reply_markup=project_menu(slug),
            )
        return

    await update.message.reply_text("از منو یک گزینه انتخاب کن.", reply_markup=main_menu())


async def on_document(update, context):
    uid = update.effective_user.id
    st = _STATE.get(uid) or {}
    store: ProjectStore = context.bot_data["store"]

    if st.get("mode") != "await_zip":
        await update.message.reply_text(
            "اول از منوی پروژه «Upload chapter ZIP» رو بزن.",
            reply_markup=main_menu(),
        )
        return

    slug = st["slug"]
    doc = update.message.document
    if not doc.file_name.lower().endswith(".zip"):
        await update.message.reply_text("فقط فایل ZIP بفرست.")
        return

    caption = (update.message.caption or "").strip() or Path(doc.file_name).stem
    await update.message.reply_text(
        f"Downloading & processing `{caption}` … این ممکنه طول بکشه (OCR+translate+clean).",
        parse_mode="Markdown",
    )

    tg_file = await context.bot.get_file(doc.file_id)
    uploads = store.path(slug) / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    zip_path = uploads / doc.file_name
    await tg_file.download_to_drive(custom_path=str(zip_path))

    try:
        result = await asyncio.to_thread(
            run_chapter_from_zip, store, slug, zip_path, caption
        )
    except Exception as e:
        logger.exception("chapter job failed")
        await update.message.reply_text(f"Failed: {e}", reply_markup=project_menu(slug))
        return

    context.user_data["last_chapter_dir"] = result["chapter_dir"]
    _STATE[uid] = {
        "mode": None,
        "slug": slug,
        "chapter_dir": result["chapter_dir"],
    }

    missing = result.get("missing_names") or []
    msg = (
        f"Done.\n"
        f"Pages: {result['pages']} | regions: {result['regions']}\n"
        f"Folder: `{result['chapter_dir']}`\n"
    )
    if missing:
        msg += "\n⚠️ این اسم‌ها تو glossary نبودن (حدسی):\n"
        msg += "\n".join(f"• {n}" for n in missing[:25])
        msg += "\n\nاگه می‌خوای درست بشن، فارسی‌شونو بده و بعد Retranslate."
        buttons = [[(f"➕ {n}", f"missing_add:{slug}:{n}")] for n in missing[:12]]
        buttons.append([("🔁 Retranslate now", f"retranslate:{slug}")])
        buttons.append([("📁 Project", f"proj_open:{slug}")])
        await update.message.reply_text(msg, reply_markup=_kb(buttons), parse_mode="Markdown")
    else:
        msg += "\nاسم مشکوکِ جاافتاده‌ای پیدا نشد."
        await update.message.reply_text(msg, reply_markup=project_menu(slug), parse_mode="Markdown")

    fa_path = result.get("scanlation_fa")
    if fa_path and Path(fa_path).exists():
        with open(fa_path, "rb") as fh:
            await update.message.reply_document(
                document=fh,
                filename="chapter_scanlation_fa.txt",
                caption="ترجمه اسکنلیشن (@ per line)",
            )
    both = result.get("scanlation_en_fa")
    if both and Path(both).exists():
        with open(both, "rb") as fh:
            await update.message.reply_document(
                document=fh,
                filename="chapter_scanlation_en_fa.txt",
            )


def run_bot(token: Optional[str] = None) -> None:
    setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "Set TELEGRAM_BOT_TOKEN in env or .env\n"
            "If you pasted a token in chat, revoke it in @BotFather first."
        )

    try:
        from telegram.ext import (
            Application,
            CallbackQueryHandler,
            CommandHandler,
            MessageHandler,
            filters,
        )
    except ImportError as e:
        raise SystemExit(
            "pip install 'python-telegram-bot>=21.0'\n"
            "Optional: pip install manga-ai[telegram]"
        ) from e

    store = ProjectStore()
    app = Application.builder().token(token).build()
    app.bot_data["store"] = store
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    logger.info("Telegram bot starting…")
    app.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    run_bot()
