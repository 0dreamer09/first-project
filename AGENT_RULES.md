# Project Rules & Guidelines: Telegram AI Bot & Dashboard

## 1. Project Overview & Architecture
- Purpose: Telegram AI Assistant integrated with Google Gemini API, managed via a real-time web dashboard.
- Backend: Python 3.11+, FastAPI, Uvicorn, Aiogram 3.x (asynchronous lifecycle).
- Frontend: Clean HTML5, Tailwind CSS (via CDN), vanilla JavaScript.
- AI Integration: Google Gemini API via official SDK (`google-genai`), using resilient fallback chains.

## 2. Hard Constraints & Coding Standards
- Aiogram Version: Always use `aiogram 3.x` syntax (`Router`, `Dispatcher`, `CommandStart()`). Never use deprecated aiogram 2.x methods.
- Security: Never hardcode API keys or bot tokens inside Python files. All secrets MUST be read from `.env` via `python-dotenv`.
- Encoding & Logs: Save all project files as UTF-8 without BOM. Console logs and terminal outputs must use plain English (ASCII) to avoid Windows PowerShell encoding glitches.
- Non-Destructive Edits: Modify only the necessary functions or blocks. Never rewrite entire working files from scratch unless explicitly instructed.

## 3. Resilience & Network Rules
- API Timeouts: All external LLM requests must have an asynchronous timeout (maximum 6.0 seconds) to trigger fallback models instantly.
- Telegram Formatting: Stream chunks as raw unparsed text (`parse_mode=None`). Apply `parse_mode="HTML"` only on the final completed message. Wrap final message edits in `try/except TelegramBadRequest`.

## 4. Git & Version Control Automation
- Commits: After successfully implementing, testing, and verifying any feature or bugfix, automatically create a concise Git commit (e.g., `feat: ...`, `fix: ...`).
- Push: Push verified changes directly to the `main` branch on GitHub.
- Ignored Files: Never commit `.env`, `*.pyc`, `__pycache__/`, `.venv/`, or local SQLite/JSON user databases.
