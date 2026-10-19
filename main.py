import asyncio
import html
import io
import json
import os
import re
import sys
import time
from typing import Optional, List, Dict, Any, Tuple
import unicodedata
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
MAX_HISTORY_MESSAGES = 10  # Хранение последних 10 сообщений (5 раундов диалога: user + model)
MAX_VOICE_SIZE_BYTES = 20 * 1024 * 1024  # 20 МБ максимальный размер голосового сообщения

# Счетчики статистики и лента активности
stats_counters: Dict[str, int] = {
    "text_requests": 0,
    "voice_requests": 0,
    "fallback_triggers": 0,
}
recent_events: List[Dict[str, Any]] = []
MAX_RECENT_EVENTS = 20

def log_event(
    chat_id: int,
    username: Optional[str],
    event_type: str,
    model: str,
    status: str,
    detail: str = ""
):
    """Добавляет событие в память (хранятся последние 20 событий)."""
    now_str = time.strftime("%H:%M:%S")
    user_label = f"@{username}" if username else f"id:{chat_id}"
    event = {
        "timestamp": now_str,
        "chat_id": chat_id,
        "username": user_label,
        "type": event_type,  # "text" или "voice"
        "model": model,
        "status": status,    # "success", "fallback", "error"
        "detail": detail
    }
    recent_events.append(event)
    if len(recent_events) > MAX_RECENT_EVENTS:
        recent_events.pop(0)

# ----------------- Персистентное хранение переписки (chats/{chat_id}.json) -----------------

CHATS_DIR = os.path.join(os.path.dirname(__file__), "chats")

def get_user_chat_file(chat_id: int) -> str:
    """Возвращает путь к файлу переписки конкретного пользователя."""
    if not os.path.exists(CHATS_DIR):
        os.makedirs(CHATS_DIR, exist_ok=True)
    return os.path.join(CHATS_DIR, f"{chat_id}.json")

def load_user_messages(chat_id: int) -> List[Dict[str, Any]]:
    """Загружает сохраненную историю сообщений пользователя из chats/{chat_id}.json."""
    path = get_user_chat_file(chat_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read chat history for {chat_id}: {e}", file=sys.stderr)
        return []

def append_user_message(chat_id: int, role: str, msg_type: str, text: str):
    """Добавляет сообщение (user или bot) в историю chats/{chat_id}.json."""
    messages = load_user_messages(chat_id)
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    messages.append({
        "timestamp": now_str,
        "role": role,        # "user" или "bot"
        "type": msg_type,    # "text" или "voice"
        "text": text
    })
    path = get_user_chat_file(chat_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save chat message for {chat_id}: {e}", file=sys.stderr)


def get_user_history(chat_id: int) -> List[genai_types.Content]:
    """
    Возвращает накопленную историю диалога пользователя (до MAX_HISTORY_MESSAGES = 10).
    Гарантирует валидность структуры для Gemini API (история всегда начинается с role='user').
    """
    raw_history = user_histories.get(chat_id, [])
    trimmed = list(raw_history[-MAX_HISTORY_MESSAGES:])
    while trimmed and trimmed[0].role != "user":
        trimmed.pop(0)
    return trimmed

def extract_transcription_from_response(text: str) -> str:
    """
    Извлекает расшифровку сказанного из ответа модели для сохранения в историю контекста.
    """
    # 1. Поиск строки с иконкой 🎤
    m = re.search(r'🎤[ \t]*(?:<i>|[«"“*]|&lt;i&gt;)*[ \t]*(.*?)[ \t]*(?:</i>|[»"”*]|&lt;/i&gt;)*(?:\n|$)', text)
    if m and m.group(1).strip():
        t = re.sub(r'<[^>]+>|&lt;[^&]+&gt;', '', m.group(1)).strip('«»"\'* \t\r\n')
        if t:
            return t
    # 2. Поиск строки вида "Расшифровка: ..."
    m = re.search(r'(?:[Рр]асшифровка|[Тт]ранскрипция):\s*[«"“]?(.*?)[»"”]?(?:\n|$)', text)
    if m and m.group(1).strip():
        t = re.sub(r'<[^>]+>|&lt;[^&]+&gt;', '', m.group(1)).strip('«»"\'* \t\r\n')
        if t:
            return t
    # 3. Резерв: первая непустая строка до 200 символов
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines and len(lines[0]) < 200:
        cleaned = re.sub(r'^[🎤*#_ \t]+', '', lines[0])
        cleaned = re.sub(r'<[^>]+>|&lt;[^&]+&gt;', '', cleaned).strip('«»"\'* \t\r\n')
        if cleaned:
            return cleaned
    return ""


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

def is_emoji_or_symbol(char: str) -> bool:
    """Проверяет, является ли символ эмодзи или специальным маркером-значком."""
    if not char:
        return False
    cat = unicodedata.category(char)
    if cat in ("So", "Sm", "Sk"):
        return True
    code = ord(char)
    if (
        code >= 0x1F000
        or (0x2000 <= code <= 0x2BFF)
        or (0xE000 <= code <= 0xF8FF)
        or (0xFE00 <= code <= 0xFE0F)
    ):
        if char not in "«»\"'()[]{}<>,.;:!?/\\|-_=+*&^%$#@~`":
            return True
    return False

def clean_list_line(line: str) -> str:
    """
    Очищает маркдаун-маркеры списка (*, - в начале строки):
    - Если после маркера идет эмодзи или значок: оставляет только эмодзи.
    - Если идет обычный текст/форматирование: заменяет на красивый маркер списка •.
    """
    m = re.match(r'^([ \t]*)[*\-][ \t]+(.*)$', line)
    if not m:
        return line
    indent = m.group(1)
    rest = m.group(2).strip()
    if not rest:
        return ""
    first_char = rest[0]
    if (
        is_emoji_or_symbol(first_char)
        or first_char in "•🔹🔸▫▪✔✅❌👉💡📌⚡🎯🚀"
        or re.match(r'^\d+[\.\)]\s+', rest)
        or re.match(r'^[•⁃‣–—]\s+', rest)
    ):
        return f"{indent}{rest}"
    else:
        return f"{indent}• {rest}"

def markdown_to_telegram_html(text: str) -> str:
    """
    Конвертирует стандартный Markdown от Gemini в валидный Telegram HTML:
    - Извлекает и экранирует блоки кода (```...```) и инлайн-код (`...`)
    - Экранирует сырые символы (&, <, >) в основном тексте
    - Восстанавливает валидные парные HTML-теги от модели (<b>, <i>, <code>, <s>)
    - Оформляет строку транскрипции голосового сообщения (🎤 <i>«...»</i>)
    - Удаляет горизонтальные разделители (***, ---, ___)
    - Очищает маркдаун-маркеры списков (* или -) перед эмодзи и текстом
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

    # 4. Восстанавливаем валидные парные HTML-теги, сгенерированные моделью
    text = re.sub(r'&lt;([bius])&gt;(.*?)&lt;/\1&gt;', r'<\1>\2</\1>', text, flags=re.DOTALL)

    # 5. Специальная обработка строки транскрипции голосового сообщения (🎤 ...)
    def format_voice_line(match):
        raw_content = match.group(1).strip()
        # Очищаем от возможных сырых или экранированных тегов
        cleaned = re.sub(r'</?(?:i|em|b|strong)>|&lt;/?(?:i|em|b|strong)&gt;', '', raw_content).strip()
        # Очищаем от обрамляющего маркдауна (*, _)
        cleaned = re.sub(r'^[*_]+|[*_]+$', '', cleaned).strip()
        # Очищаем от внешних кавычек любого вида
        cleaned = re.sub(r'^(?:«|&quot;|"|\'|“)+|(?:»|&quot;|"|\'|”)+$', '', cleaned).strip()
        if not cleaned:
            return match.group(0)
        return f"🎤 <i>«{cleaned}»</i>"

    text = re.sub(r'^[ \t]*🎤[ \t]*(.+)$', format_voice_line, text, flags=re.MULTILINE)

    # 6. Удаляем горизонтальные разделители (***, ---, ___)
    text = re.sub(r'^[ \t]*([*\-_]){3,}[ \t]*$', '', text, flags=re.MULTILINE)

    # 7. Очистка маркдаун-маркеров списка (* или - в начале строк)
    lines = text.split('\n')
    text = '\n'.join(clean_list_line(line) for line in lines)

    # 8. Заголовки (###, ##, #) -> <b>...</b> с отступом
    text = re.sub(r'^(#{1,6})\s+(.+)$', r'\n<b>\2</b>\n', text, flags=re.MULTILINE)

    # 9. Жирный текст (**текст** или __текст__)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text, flags=re.DOTALL)

    # 10. Курсив (*текст* или _текст_)
    text = re.sub(r'(?<!\w)\*([^*]+?)\*(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!\w)_([^_]+?)_(?!\w)', r'<i>\1</i>', text)

    # 11. Возвращаем блоки кода и инлайн-код
    for idx, block in enumerate(code_blocks):
        text = text.replace(f'@@CODEBLOCK_{idx}@@', block)
    for idx, inline in enumerate(inline_codes):
        text = text.replace(f'@@INLINECODE_{idx}@@', inline)

    # Чистим избыточные переносы строк
    return re.sub(r'\n{3,}', '\n\n', text).strip()

# ----------------- Fallback Chain со стримингом и таймаутом 6 секунд -----------------

async def generate_and_stream_response(
    chat_id: int,
    user_text: Optional[str],
    system_instruction: str,
    user_tag: str,
    status_msg: types.Message,
    audio_bytes: Optional[bytes] = None,
    audio_mime_type: str = "audio/ogg",
    username: Optional[str] = None
) -> bool:
    """
    Каскадный опрос моделей со стримингом (Streaming response) и таймаутом 6.0 секунд на каждую модель.
    Поддерживает как текстовые сообщения, так и голосовые сообщения (мультимодальное аудио).
    Обновляет status_msg чанками каждые 1–1.5 секунды как обычный текст (parse_mode=None).
    Финальный ответ конвертируется в валидный Telegram HTML с откатом на чистый текст.
    """
    client = get_or_init_gemini_client()
    if not client:
        return False

    # Учитываем счетчик запросов (текст или голос)
    req_type = "voice" if audio_bytes else "text"
    if req_type == "voice":
        stats_counters["voice_requests"] += 1
    else:
        stats_counters["text_requests"] += 1

    cfg = load_config()
    primary_model = cfg.get("model", "gemini-3.5-flash")
    configured_fallbacks = cfg.get("fallback_models", DEFAULT_CONFIG["fallback_models"])

    # Цепочка опроса: приоритетная модель первой, затем запасные
    models_chain = [primary_model] + [m for m in configured_fallbacks if m != primary_model]

    # Формируем части контента для текущего запроса (текст или аудио + инструкция)
    if audio_bytes:
        audio_part = genai_types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime_type)
        voice_prompt = (
            "Прослушай приложенное голосовое сообщение.\n"
            "1. В самой первой строке обязательно выведи точную расшифровку сказанного (транскрипцию) в формате:\n"
            "🎤 <i>«расшифрованный текст»</i>\n\n"
            "2. Затем сделай пустую строку и дай развернутый, точный и полезный ответ на вопрос или сообщение "
            "с учётом системной инструкции и контекста предыдущего диалога."
        )
        current_turn_parts = [audio_part, genai_types.Part.from_text(text=voice_prompt)]
    else:
        current_turn_parts = [genai_types.Part.from_text(text=user_text or "")]

    # Подготавливаем историю диалога для контекста (до 10 сообщений / 5 раундов)
    history = get_user_history(chat_id)
    request_contents = list(history) + [
        genai_types.Content(
            role="user",
            parts=current_turn_parts
        )
    ]

    total_models = len(models_chain)
    input_desc = f"audio ({len(audio_bytes)} bytes)" if audio_bytes else f"text ({len(user_text or '')} chars)"

    for idx, model_name in enumerate(models_chain):
        is_fallback = (idx > 0)
        next_model = models_chain[idx + 1] if idx + 1 < total_models else None

        try:
            print(f"[Gemini -> {user_tag}] Requesting model '{model_name}' (attempt {idx + 1}/{total_models}, timeout 6s, {input_desc}, context: {len(history)} msgs)...")
            
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
                print(f"[Warning] Failed to parse Telegram HTML ({e}), sending plain text...")
                plain_final = accumulated_text
                if is_fallback:
                    plain_final += f"\n\n(Резервная модель: {model_name})"
                try:
                    await status_msg.edit_text(plain_final, parse_mode=None)
                except TelegramAPIError:
                    pass

            # Сохраняем диалог в истории (до MAX_HISTORY_MESSAGES = 10 сообщений / 5 раундов)
            if accumulated_text.strip():
                if chat_id not in user_histories:
                    user_histories[chat_id] = []

                if audio_bytes:
                    transcription = extract_transcription_from_response(accumulated_text)
                    if transcription:
                        saved_user_text = f"🎤 [Голосовое сообщение]: {transcription}"
                        chat_user_text = transcription
                    else:
                        saved_user_text = "🎤 [Голосовое сообщение]"
                        chat_user_text = "[Голосовое сообщение]"
                    msg_type = "voice"
                else:
                    saved_user_text = user_text or ""
                    chat_user_text = user_text or ""
                    msg_type = "text"

                user_histories[chat_id].append(
                    genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=saved_user_text)])
                )
                user_histories[chat_id].append(
                    genai_types.Content(role="model", parts=[genai_types.Part.from_text(text=accumulated_text)])
                )
                user_histories[chat_id] = user_histories[chat_id][-MAX_HISTORY_MESSAGES:]

                # Персистентное сохранение реплик в файл переписки chats/{chat_id}.json
                append_user_message(chat_id=chat_id, role="user", msg_type=msg_type, text=chat_user_text)
                append_user_message(chat_id=chat_id, role="bot", msg_type="text", text=accumulated_text)

            # Логируем событие в ленту активности
            status_label = "fallback" if is_fallback else "success"
            log_event(
                chat_id=chat_id,
                username=username,
                event_type=req_type,
                model=model_name,
                status=status_label,
                detail=f"{len(accumulated_text)} chars"
            )

            print(f"[Gemini OK -> {user_tag}] Response delivered from '{model_name}' ({len(accumulated_text)} chars, stored history: {len(user_histories[chat_id])} msgs)")
            return True

        except (asyncio.TimeoutError, TimeoutError):
            stats_counters["fallback_triggers"] += 1
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
            stats_counters["fallback_triggers"] += 1
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

    log_event(
        chat_id=chat_id,
        username=username,
        event_type=req_type,
        model=models_chain[0] if models_chain else "unknown",
        status="error",
        detail="All models exhausted"
    )
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
        f"💬 Напишите мне вопрос текстом или отправьте <b>голосовое сообщение</b> — я расшифрую его и отвечу.\n"
        f"🔄 Команда /reset — очистить историю диалога.\n"
        f"🌐 Веб-панель: <code>http://localhost:{WEB_PORT}</code>",
        parse_mode=ParseMode.HTML
    )

@dp.message(Command("reset", "clear"))
async def command_reset_handler(message: types.Message):
    chat_id = message.chat.id
    had_history = chat_id in user_histories and bool(user_histories[chat_id])
    user_histories.pop(chat_id, None)
    print(f"[Bot /reset] Conversation history completely cleared for chat_id: {chat_id} (had previous context: {had_history})")
    await message.answer(
        "🔄 <b>История диалога очищена.</b>\n"
        "Контекст сброшен, можем начать общение с чистого листа!",
        parse_mode=ParseMode.HTML
    )

@dp.message(F.voice)
async def ai_voice_handler(message: types.Message):
    """
    Обработчик голосовых сообщений:
    - Проверяет размер (до 20 МБ)
    - Скачивает аудио (.ogg Opus) в память через bot.download()
    - Передает в Gemini со стримингом, Fallback Chain и контекстной памятью
    - Сохраняет транскрипцию в историю пользователя
    """
    chat_id = message.chat.id
    username = message.from_user.username if message.from_user else None
    user_tag = f"@{username}" if username else f"id:{chat_id}"
    full_name = message.from_user.full_name if message.from_user else "пользователь"

    save_user(chat_id=chat_id, username=username, full_name=full_name)

    client = get_or_init_gemini_client()
    if not client:
        await message.answer(
            "⚠️ <b>GEMINI_API_KEY</b> не настроен в файле <code>.env</code>.\n"
            "Добавьте ключ в <code>.env</code>, чтобы включить распознавание голосовых сообщений.",
            parse_mode=ParseMode.HTML
        )
        return

    # Проверка размера голосового сообщения
    file_size = message.voice.file_size or 0
    if file_size > MAX_VOICE_SIZE_BYTES:
        await message.answer(
            "⚠️ <b>Голосовое сообщение слишком большое</b> (превышает 20 МБ).\n"
            "Пожалуйста, отправьте более короткую запись.",
            parse_mode=ParseMode.HTML
        )
        return

    # Отправляем индикатор набора текста
    await message.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    status_msg = await message.answer("🎧 Слушаю голосовое сообщение...")

    try:
        # Скачиваем голосовое сообщение в память
        voice_buffer = io.BytesIO()
        await message.bot.download(message.voice, destination=voice_buffer)
        audio_bytes = voice_buffer.getvalue()

        if not audio_bytes:
            raise ValueError("Audio buffer is empty")

        if len(audio_bytes) > MAX_VOICE_SIZE_BYTES:
            await status_msg.edit_text(
                "⚠️ Голосовое сообщение превышает лимит в 20 МБ. Пожалуйста, отправьте более короткую запись."
            )
            return

        print(f"[Voice -> {user_tag}] Downloaded voice note: {len(audio_bytes)} bytes, duration: {message.voice.duration}s")

    except Exception as e:
        print(f"[Voice ERROR -> {user_tag}] Failed to download audio: {e}")
        try:
            await status_msg.edit_text(
                "❌ Не удалось обработать аудиофайл (возможно, файл поврежден). "
                "Попробуйте записать голосовое сообщение ещё раз или написать текстом."
            )
        except TelegramAPIError:
            pass
        return

    cfg = load_config()
    system_instruction = cfg.get("system_instruction", "")

    # Запускаем распознавание и ответ через Gemini со стримингом и цепочкой fallback
    await generate_and_stream_response(
        chat_id=chat_id,
        user_text=None,
        system_instruction=system_instruction,
        user_tag=user_tag,
        status_msg=status_msg,
        audio_bytes=audio_bytes,
        audio_mime_type=message.voice.mime_type or "audio/ogg",
        username=username
    )

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
        status_msg=status_msg,
        username=username
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
    """Возвращает расширенную статистику и недавние события для панели управления."""
    users = load_users()
    cfg = load_config()
    has_gemini = bool(os.getenv("GEMINI_API_KEY", "").strip() or gemini_client)
    return {
        "user_count": len(users),
        "bot_configured": bool(BOT_TOKEN),
        "gemini_configured": has_gemini,
        "model": cfg.get("model", "gemini-3.5-flash"),
        "fallback_models": cfg.get("fallback_models", DEFAULT_CONFIG["fallback_models"]),
        "active_dialogues": len(user_histories),
        "text_requests": stats_counters["text_requests"],
        "voice_requests": stats_counters["voice_requests"],
        "fallback_triggers": stats_counters["fallback_triggers"],
        "recent_events": list(reversed(recent_events))
    }

# ----------------- Работа с пользователями и историей переписки -----------------

@app.get("/api/users")
async def get_users_list_endpoint():
    """
    Возвращает список всех зарегистрированных пользователей со статистикой переписки:
    chat_id, username, first_name, last_message_time, message_count.
    """
    users = load_users()
    user_list = []
    for u in users:
        chat_id = u["chat_id"]
        messages = load_user_messages(chat_id)
        last_time = messages[-1]["timestamp"] if messages else None
        full_name = (u.get("full_name") or "").strip()
        first_name = full_name.split()[0] if full_name else (u.get("username") or f"id:{chat_id}")
        user_list.append({
            "chat_id": chat_id,
            "username": u.get("username"),
            "first_name": first_name,
            "full_name": full_name,
            "last_message_time": last_time,
            "message_count": len(messages)
        })
    # Сортируем: сначала активные пользователи с самыми свежими сообщениями
    user_list.sort(key=lambda x: x["last_message_time"] or "", reverse=True)
    return user_list

@app.get("/api/users/{chat_id}/history")
async def get_user_chat_history_endpoint(chat_id: int):
    """
    Возвращает полную хронологическую историю сообщений пользователя.
    """
    messages = load_user_messages(chat_id)
    return {
        "chat_id": chat_id,
        "message_count": len(messages),
        "messages": messages
    }

class DirectMessageRequest(BaseModel):
    message: str

@app.post("/api/users/{chat_id}/send")
async def send_direct_message_endpoint(chat_id: int, req: DirectMessageRequest):
    """
    Отправляет прямое текстовое сообщение в Telegram конкретному пользователю
    и сохраняет его в историю переписки.
    """
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Текст сообщения не может быть пустым")

    if not bot_instance or not BOT_TOKEN:
        raise HTTPException(status_code=503, detail="Telegram-бот не запущен или BOT_TOKEN не настроен")

    try:
        await bot_instance.send_message(chat_id=chat_id, text=text)
        # Сохраняем отправленное ботом сообщение в историю
        append_user_message(chat_id=chat_id, role="bot", msg_type="text", text=text)
        print(f"[Direct Message -> chat_id:{chat_id}] Sent: \"{text}\"")
        return {"status": "success", "detail": "Сообщение успешно доставлено"}
    except TelegramAPIError as e:
        print(f"[Direct Message ERROR -> chat_id:{chat_id}] Telegram API error: {e.message}")
        raise HTTPException(status_code=500, detail=f"Ошибка Telegram API: {e.message}")
    except Exception as e:
        print(f"[Direct Message ERROR -> chat_id:{chat_id}] Send error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    print(f"[AI Config] Configuration updated: model={cfg.get('model')}, Fallback Chain={cfg.get('fallback_models')}, user history reset.")

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
    print("[Aiogram] Starting Telegram bot (polling)...")
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await bot.set_my_commands([
            types.BotCommand(command="start", description="Запустить бота"),
            types.BotCommand(command="reset", description="Очистить историю диалога"),
        ])
        await dp.start_polling(bot)
    except Exception as e:
        print(f"[Aiogram ERROR] Error running bot: {e}", file=sys.stderr)

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
