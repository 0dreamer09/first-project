import asyncio
import html
import json
import os
import re
import sys
import time
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel

# Обеспечиваем корректный вывод UTF-8 в консоли Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ChatAction
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest

from google import genai
from google.genai import types as genai_types

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# Загрузка переменных окружения из .env
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
WEB_PORT = int(os.getenv("WEB_PORT", "8000"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

# Глобальные объекты
bot_instance: Optional[Bot] = None
gemini_client: Optional[genai.Client] = None
user_histories: Dict[int, List[genai_types.Content]] = {}

# ----------------- Работа с конфигурацией AI (config.json) -----------------

DEFAULT_CONFIG = {
    "system_instruction": "Ты — полезный и лаконичный AI-ассистент в Telegram. Отвечай дружелюбно, структурированно и по делу.",
    "model": "gemini-3.5-flash",
    "fallback_models": [
        "gemini-3.5-flash",
        "gemini-3.8-flash",
        "gemini-3.1-flash-lite",
        "gemini-3-flash-preview"
    ]
}

def load_config() -> Dict[str, Any]:
    """Загружает конфигурацию из config.json."""
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "fallback_models" not in data:
                data["fallback_models"] = DEFAULT_CONFIG["fallback_models"]
            return data
    except Exception as e:
        print(f"[ERROR] Ошибка чтения config.json: {e}", file=sys.stderr)
        return DEFAULT_CONFIG.copy()

def save_config(cfg: Dict[str, Any]):
    """Сохраняет конфигурацию в config.json."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] Ошибка записи в config.json: {e}", file=sys.stderr)

def get_or_init_gemini_client() -> Optional[genai.Client]:
    """Получает или динамически инициализирует клиент Google Gemini."""
    global gemini_client
    if gemini_client:
        return gemini_client
    
    load_dotenv(override=True)
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        try:
            gemini_client = genai.Client(
                api_key=key,
                http_options=genai_types.HttpOptions(
                    retry_options=genai_types.HttpRetryOptions(attempts=1)
                )
            )
            print("[Gemini] Клиент Gemini API успешно подключен.")
            return gemini_client
        except Exception as e:
            print(f"[Gemini ERROR] Не удалось инициализировать клиент Gemini: {e}", file=sys.stderr)
    return None

# ----------------- Форматирование текста (Markdown -> Telegram HTML) -----------------

def markdown_to_telegram_html(text: str) -> str:
    """
    Конвертирует стандартный Markdown от Gemini в валидный Telegram HTML:
    - Извлекает и экранирует блоки кода (```...```) и инлайн-код (`...`)
    - Экранирует сырые символы (&, <, >) в основном тексте
    - Удаляет горизонтальные разделители (***, ---, ___)
    - Заменяет заголовки (###, ##, #) на <b>...</b> с отступом
    - Заменяет **текст** / __текст__ на <b>текст</b> (жирный)
    - Заменяет *текст* / _текст_ на <i>текст</i> (курсив)
    - Возвращает защищенный код в тегах <pre><code> и <code>
    """
    code_blocks = []
    inline_codes = []

    def save_code_block(match):
        lang = match.group(1).strip() if match.group(1) else ''
        code = match.group(2)
        escaped_code = html.escape(code)
        idx = len(code_blocks)
        if lang:
            code_blocks.append(f'<pre><code class="language-{lang}">{escaped_code}</code></pre>')
        else:
            code_blocks.append(f'<pre><code>{escaped_code}</code></pre>')
        return f'@@CODEBLOCK_{idx}@@'

    def save_inline_code(match):
        code = match.group(1)
        escaped_code = html.escape(code)
        idx = len(inline_codes)
        inline_codes.append(f'<code>{escaped_code}</code>')
        return f'@@INLINECODE_{idx}@@'

    # 1. Извлекаем блоки кода
    text = re.sub(r'```([a-zA-Z0-9_+-]*)\n?(.*?)```', save_code_block, text, flags=re.DOTALL)

    # 2. Извлекаем инлайн-код
    text = re.sub(r'`([^`\n]+)`', save_inline_code, text)

    # 3. Экранируем сырые символы в основном тексте (&, <, >)
    text = html.escape(text)

    # 4. Удаляем горизонтальные разделители (***, ---, ___)
    text = re.sub(r'^[ \t]*([*\-_]){3,}[ \t]*$', '', text, flags=re.MULTILINE)

    # 5. Заголовки (###, ##, #) -> <b>...</b> с отступом
    text = re.sub(r'^(#{1,6})\s+(.+)$', r'\n<b>\2</b>\n', text, flags=re.MULTILINE)

    # 6. Жирный текст (**текст** или __текст__)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text, flags=re.DOTALL)

    # 7. Курсив (*текст* или _текст_)
    text = re.sub(r'(?<!\w)\*([^*]+?)\*(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!\w)_([^_]+?)_(?!\w)', r'<i>\1</i>', text)

    # 8. Возвращаем блоки кода и инлайн-код
    for idx, block in enumerate(code_blocks):
        text = text.replace(f'@@CODEBLOCK_{idx}@@', block)
    for idx, inline in enumerate(inline_codes):
        text = text.replace(f'@@INLINECODE_{idx}@@', inline)

    # Чистим избыточные переносы строк
    return re.sub(r'\n{3,}', '\n\n', text).strip()

# ----------------- Fallback Chain со стримингом и таймаутом 6 секунд -----------------

async def generate_and_stream_response(
    chat_id: int,
    user_text: str,
    system_instruction: str,
    user_tag: str,
    status_msg: types.Message
) -> bool:
    """
    Каскадный опрос моделей со стримингом (Streaming response) и таймаутом 6.0 секунд на каждую модель.
    Обновляет status_msg чанками каждые 1–1.5 секунды как обычный текст (parse_mode=None).
    Финальный ответ конвертируется в валидный Telegram HTML с откатом на чистый текст.
    """
    client = get_or_init_gemini_client()
    if not client:
        return False

    cfg = load_config()
    primary_model = cfg.get("model", "gemini-3.5-flash")
    configured_fallbacks = cfg.get("fallback_models", DEFAULT_CONFIG["fallback_models"])

    # Цепочка опроса: приоритетная модель первой, затем запасные
    models_chain = [primary_model] + [m for m in configured_fallbacks if m != primary_model]

    # Подготавливаем историю диалога для контекста
    history = user_histories.get(chat_id, [])
    request_contents = list(history) + [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=user_text)]
        )
    ]

    total_models = len(models_chain)

    for idx, model_name in enumerate(models_chain):
        is_fallback = (idx > 0)
        next_model = models_chain[idx + 1] if idx + 1 < total_models else None

        try:
            print(f"[Gemini -> {user_tag}] Запрос к модели '{model_name}' (попытка {idx + 1}/{total_models}, таймаут 6s)...")
            
            # Запускаем стриминг с жестким таймаутом 6.0 секунд на получение первого ответа
            stream = await asyncio.wait_for(
                client.aio.models.generate_content_stream(
                    model=model_name,
                    contents=request_contents,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_instruction
                    )
                ),
                timeout=6.0
            )

            # Получаем первый чанк не позднее 6.0 секунд
            first_chunk = await asyncio.wait_for(stream.__anext__(), timeout=6.0)
            accumulated_text = first_chunk.text or ""

            last_edit_time = time.time()
            last_displayed_text = ""

            # Первое отображение ответа пользователю (обычный текст без parse_mode)
            if accumulated_text.strip():
                try:
                    await status_msg.edit_text(accumulated_text + " ▍", parse_mode=None)
                    last_displayed_text = accumulated_text
                    last_edit_time = time.time()
                except TelegramAPIError:
                    pass

            # Читаем последующие чанки потока
            async for chunk in stream:
                piece = chunk.text or ""
                if not piece:
                    continue
                accumulated_text += piece

                # Обновляем текст чанками каждые 1–1.5 секунды как обычный текст (parse_mode=None)
                now = time.time()
                if (now - last_edit_time >= 1.2) and (accumulated_text != last_displayed_text):
                    try:
                        await status_msg.edit_text(accumulated_text + " ▍", parse_mode=None)
                        last_displayed_text = accumulated_text
                        last_edit_time = now
                    except TelegramAPIError:
                        pass

            # Финальное форматирование в валидный Telegram HTML
            final_html = markdown_to_telegram_html(accumulated_text)
            if is_fallback:
                final_html += f"\n\n<i>⚡ Ответ сгенерирован резервной моделью: {model_name}</i>"

            try:
                await status_msg.edit_text(final_html, parse_mode=ParseMode.HTML)
            except (TelegramBadRequest, TelegramAPIError) as e:
                print(f"[Warning] Ошибка парсинга Telegram HTML ({e}), отправка чистым текстом...")
                plain_final = accumulated_text
                if is_fallback:
                    plain_final += f"\n\n(Резервная модель: {model_name})"
                try:
                    await status_msg.edit_text(plain_final, parse_mode=None)
                except TelegramAPIError:
                    pass

            # Сохраняем диалог в истории
            if chat_id not in user_histories:
                user_histories[chat_id] = []
            user_histories[chat_id].append(
                genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=user_text)])
            )
            user_histories[chat_id].append(
                genai_types.Content(role="model", parts=[genai_types.Part.from_text(text=accumulated_text)])
            )
            user_histories[chat_id] = user_histories[chat_id][-20:]

            print(f"[Gemini OK -> {user_tag}] Ответ успешно доставлен от '{model_name}' ({len(accumulated_text)} симв.)")
            return True

        except (asyncio.TimeoutError, TimeoutError):
            error_str = "Timeout 6.0s (модель не ответила за 6 сек)"
            if next_model:
                print(f"[Warning] Model {model_name} failed ({error_str}), trying fallback to {next_model}...")
                try:
                    await status_msg.edit_text(f"💭 Переключаюсь на {next_model}...")
                except TelegramAPIError:
                    pass
                await asyncio.sleep(1)
            else:
                print(f"[Warning] Model {model_name} failed ({error_str}), all fallback models exhausted.")

        except Exception as e:
            error_str = str(e)
            if next_model:
                print(f"[Warning] Model {model_name} failed ({error_str}), trying fallback to {next_model}...")
                try:
                    await status_msg.edit_text(f"💭 Переключаюсь на {next_model}...")
                except TelegramAPIError:
                    pass
                await asyncio.sleep(1)
            else:
                print(f"[Warning] Model {model_name} failed ({error_str}), all fallback models exhausted.")

    # Если абсолютно все модели из цепочки упали
    try:
        await status_msg.edit_text("Все сервера Gemini сейчас временно перегружены. Попробуйте через минуту.")
    except TelegramAPIError:
        pass

    return False

# ----------------- Работа со списком пользователей -----------------

def load_users() -> List[Dict[str, Any]]:
    """Загружает список сохранённых пользователей из users.json."""
    if not os.path.exists(USERS_FILE):
        return []
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Ошибка чтения {USERS_FILE}: {e}", file=sys.stderr)
        return []

def save_user(chat_id: int, username: Optional[str], full_name: Optional[str]) -> bool:
    """Сохраняет или обновляет данные пользователя в users.json."""
    users = load_users()
    existing = next((u for u in users if u["chat_id"] == chat_id), None)
    is_new = False
    
    if existing:
        existing["username"] = username
        existing["full_name"] = full_name
    else:
        users.append({
            "chat_id": chat_id,
            "username": username,
            "full_name": full_name
        })
        is_new = True

    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        return is_new
    except Exception as e:
        print(f"[ERROR] Ошибка записи в {USERS_FILE}: {e}", file=sys.stderr)
        return False

# ----------------- Aiogram 3 (Telegram-бот) -----------------

dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: types.Message):
    chat_id = message.chat.id
    username = message.from_user.username if message.from_user else None
    full_name = message.from_user.full_name if message.from_user else "пользователь"

    is_new = save_user(chat_id=chat_id, username=username, full_name=full_name)
    user_tag = f"@{username}" if username else f"chat_id:{chat_id}"

    if is_new:
        print(f"[Бот /start] Новый подписчик: {full_name} ({user_tag}, id: {chat_id}) добавлен в users.json")
    else:
        print(f"[Бот /start] Подписчик обновил данные: {full_name} ({user_tag}, id: {chat_id})")

    cfg = load_config()
    model_name = cfg.get("model", "gemini-3.5-flash")

    await message.answer(
        f"👋 Привет, <b>{full_name}</b>!\n\n"
        f"Я быстрый AI-ассистент (<b>{model_name}</b>) со стримингом ответов и каскадной защитой от перегрузок.\n\n"
        f"💬 Напишите мне любой вопрос, и я начну отвечать в реальном времени.\n"
        f"🔄 Команда /reset — очистить историю диалога.\n"
        f"🌐 Веб-панель: <code>http://localhost:{WEB_PORT}</code>",
        parse_mode=ParseMode.HTML
    )

@dp.message(Command("reset"))
async def command_reset_handler(message: types.Message):
    chat_id = message.chat.id
    if chat_id in user_histories:
        user_histories.pop(chat_id, None)
    print(f"[Бот /reset] История диалога очищена для chat_id: {chat_id}")
    await message.answer("🔄 История диалога очищена. Можем начать общение с чистого листа!")

@dp.message(F.text & ~F.text.startswith("/"))
async def ai_message_handler(message: types.Message):
    chat_id = message.chat.id
    user_text = message.text.strip()
    username = message.from_user.username if message.from_user else None
    user_tag = f"@{username}" if username else f"id:{chat_id}"

    client = get_or_init_gemini_client()
    if not client:
        await message.answer(
            "⚠️ <b>GEMINI_API_KEY</b> не настроен в файле <code>.env</code>.\n"
            "Добавьте ключ в <code>.env</code>, чтобы включить ответы искусственного интеллекта.",
            parse_mode=ParseMode.HTML
        )
        return

    # Отправляем действие ChatAction.TYPING
    await message.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # Отправляем начальное сообщение «Думаю...»
    status_msg = await message.answer("💭 Думаю...")

    cfg = load_config()
    system_instruction = cfg.get("system_instruction", "")

    # Запускаем стриминг ответа с Fallback Chain и таймаутом 6 секунд
    await generate_and_stream_response(
        chat_id=chat_id,
        user_text=user_text,
        system_instruction=system_instruction,
        user_tag=user_tag,
        status_msg=status_msg
    )

# ----------------- FastAPI веб-сервер -----------------

app = FastAPI(title="Telegram Bot Web Dashboard")

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def serve_dashboard():
    dashboard_path = os.path.join(static_dir, "dashboard.html")
    if not os.path.exists(dashboard_path):
        raise HTTPException(status_code=404, detail="dashboard.html не найден")
    return FileResponse(dashboard_path)

@app.get("/api/stats")
async def get_stats():
    """Возвращает статистику для панели управления."""
    users = load_users()
    cfg = load_config()
    has_gemini = bool(os.getenv("GEMINI_API_KEY", "").strip() or gemini_client)
    return {
        "user_count": len(users),
        "bot_configured": bool(BOT_TOKEN),
        "gemini_configured": has_gemini,
        "model": cfg.get("model", "gemini-3.5-flash"),
        "fallback_models": cfg.get("fallback_models", DEFAULT_CONFIG["fallback_models"])
    }

@app.get("/api/config")
async def get_config_endpoint():
    """Возвращает текущую конфигурацию AI."""
    return load_config()

class ConfigUpdateRequest(BaseModel):
    system_instruction: str
    model: Optional[str] = None
    fallback_models: Optional[List[str]] = None

@app.post("/api/config")
async def update_config_endpoint(req: ConfigUpdateRequest):
    """Обновляет system_instruction, model и fallback_models в config.json без перезапуска сервера."""
    instruction = req.system_instruction.strip()
    if not instruction:
        raise HTTPException(status_code=400, detail="Системная инструкция не может быть пустой")
    
    cfg = load_config()
    cfg["system_instruction"] = instruction
    if req.model and req.model.strip():
        cfg["model"] = req.model.strip()
    if req.fallback_models:
        cfg["fallback_models"] = req.fallback_models
    save_config(cfg)
    
    # Сбрасываем историю диалогов при смене системной инструкции/модели
    user_histories.clear()
    print(f"[Настройки AI] Конфигурация обновлена: модель={cfg.get('model')}, Fallback Chain={cfg.get('fallback_models')}")

    return {
        "status": "success",
        "detail": "Настройки AI успешно обновлены!",
        "config": cfg
    }

class BroadcastRequest(BaseModel):
    message: str

@app.post("/api/broadcast")
async def handle_broadcast(req: BroadcastRequest):
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Текст сообщения не может быть пустым")

    if not bot_instance or not BOT_TOKEN:
        print("[Рассылка] Ошибка: бот не инициализирован (проверьте BOT_TOKEN в .env)")
        return {
            "status": "error",
            "detail": "Токен бота не настроен в .env или бот не запущен."
        }

    users = load_users()
    if not users:
        print("[Рассылка] Запрос отклонен: список пользователей пуст.")
        return {
            "status": "warning",
            "detail": "Сначала напишите /start боту в Telegram",
            "sent_count": 0
        }

    print("\n" + "=" * 55)
    print(f"[Рассылка] Запуск рассылки текста: \"{text}\"")
    print(f"[Рассылка] Всего адресатов в базе: {len(users)}")
    print("=" * 55)

    sent_count = 0
    failed_count = 0

    for user in users:
        chat_id = user.get("chat_id")
        username = user.get("username")
        user_tag = f"@{username}" if username else f"id:{chat_id}"

        print(f"[Рассылка -> {user_tag}] Отправка: \"{text}\"...")
        try:
            await bot_instance.send_message(chat_id=chat_id, text=text)
            sent_count += 1
            print(f"[Рассылка -> {user_tag}] Успешно доставлено.")
        except TelegramAPIError as e:
            failed_count += 1
            print(f"[Рассылка -> {user_tag}] Ошибка Telegram API: {e.message}")
        except Exception as e:
            failed_count += 1
            print(f"[Рассылка -> {user_tag}] Ошибка отправки: {e}")

    print("=" * 55)
    print(f"[Рассылка] Итог: успешно доставлено {sent_count}, ошибок {failed_count}")
    print("=" * 55 + "\n")

    return {
        "status": "success",
        "detail": f"Отправлено {sent_count} пользователям",
        "sent_count": sent_count,
        "failed_count": failed_count
    }

async def run_bot(bot: Bot):
    print("[Aiogram] Запуск Telegram-бота (polling)...")
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    except Exception as e:
        print(f"[Aiogram ERROR] Ошибка при работе бота: {e}", file=sys.stderr)

async def run_web():
    print(f"[FastAPI] Запуск веб-сервера на http://localhost:{WEB_PORT} ...")
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=WEB_PORT,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    global bot_instance

    # Инициализация Gemini клиента, если ключ присутствует
    client = get_or_init_gemini_client()
    if not client:
        print("\n" + "-" * 60)
        print("[ИНФО] GEMINI_API_KEY пока не указан в .env.")
        print("       Укажите ключ в файле .env для включения AI-ответов.")
        print("-" * 60 + "\n")
    else:
        cfg = load_config()
        print(f"[Gemini] Задействована цепочка резервирования (Fallback Chain): {' -> '.join(cfg.get('fallback_models', []))}")

    tasks = [asyncio.create_task(run_web())]

    if BOT_TOKEN:
        bot_instance = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
        tasks.append(asyncio.create_task(run_bot(bot_instance)))
    else:
        print("\n" + "=" * 60)
        print("[ВНИМАНИЕ] BOT_TOKEN не указан в файле .env!")
        print("Веб-панель доступна по адресу http://localhost:8000")
        print("Для подключения бота укажите токен в файле .env.")
        print("=" * 60 + "\n")

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\nПриложение остановлено.")
