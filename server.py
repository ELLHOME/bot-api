"""
ELLHOME — бэкенд AI-консультанта для сайта / site AI-consultant backend
======================================================================
Тот же «мозг», что и в домашнем боте (RAG + Router + Tool Calling + Memory +
опц. Judge), но отдаёт JSON по HTTP, чтобы к нему обращался чат-виджет на сайте.
Ключ модели живёт ТОЛЬКО здесь, на сервере — в код сайта он не попадает.

The same "brain" as the practice bot, but served as a JSON HTTP API so the chat
widget on the site can call it. The model key lives ONLY here, on the server.

Локально / locally:   PROVIDER=ollama  uvicorn server:app --reload
Деплой / deploy:      см. README.md (Render) — PROVIDER=gemini + GOOGLE_API_KEY
"""

import os
import json
import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── Настройки / Settings ─────────────────────────────────────────────
# PROVIDER: "gemini" (облако, для деплоя) | "ollama" (локально, бесплатно)
PROVIDER = os.getenv("PROVIDER", "gemini").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "muse-glimmer")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
USE_JUDGE = os.getenv("USE_JUDGE", "false").lower() == "true"

# RU: домены, которым разрешено обращаться к API (CORS). Добавь свой прод-домен.
# EN: origins allowed to call the API (CORS). Add your production domain.
ALLOWED_ORIGINS = [
    "https://ellhome.github.io",   # прод-сайт / production site
    "http://localhost:5173",       # Vite dev
    "http://localhost:4173",       # Vite preview
    "http://127.0.0.1:5173",
]
# RU: можно переопределить через переменную окружения (через запятую)
extra = os.getenv("ALLOWED_ORIGINS")
if extra:
    ALLOWED_ORIGINS += [o.strip() for o in extra.split(",") if o.strip()]

# RU: база знаний (RAG) — факты рядом со скриптом
# EN: knowledge base (RAG) — facts next to the script
KNOWLEDGE = open(os.path.join(os.path.dirname(__file__), "knowledge.txt"), encoding="utf-8").read()

LEADS_FILE = os.path.join(os.path.dirname(__file__), "leads.json")
DATABASE_URL = os.getenv("DATABASE_URL")  # задаётся хостингом при подключении Postgres


# ── Вызов модели / Model call ────────────────────────────────────────
def call_model(messages: list[dict]) -> str:
    try:
        if PROVIDER == "ollama":
            import ollama  # RU: ленивый импорт — на сервере ollama не нужен
            return ollama.chat(model=OLLAMA_MODEL, messages=messages).message.content

        # cloud — Gemini через OpenAI-совместимый endpoint
        from openai import OpenAI
        key = os.getenv("GOOGLE_API_KEY")
        if not key or "..." in key:
            return "⚠️ На сервере не задан GOOGLE_API_KEY (нужен для PROVIDER='gemini')."
        client = OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        return client.chat.completions.create(
            model=GEMINI_MODEL, messages=messages
        ).choices[0].message.content
    except Exception as e:
        return f"⚠️ Модель не отвечает (PROVIDER={PROVIDER}): {e}"


def ask(prompt: str, system: str = "Ты — помощник.") -> str:
    return call_model([
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ])


# ── ROUTER ───────────────────────────────────────────────────────────
def classify(message: str) -> str:
    cat = ask(
        "Определи тип вопроса ОДНИМ словом из списка: цена, заявка, общее. "
        "«цена» — сколько стоит / сроки / смета. «заявка» — хочет заказать, "
        "оставить контакт, начать проект. «общее» — всё остальное. "
        f"Верни только слово.\n\nВопрос: {message}"
    ).strip().lower()
    for key in ("цена", "заявка", "общее"):
        if key in cat:
            return key
    return "общее"


# ── TOOL CALLING — сохранить заявку / save the request ───────────────
def _save_lead_to_file(name: str, service: str) -> int:
    leads = []
    if os.path.exists(LEADS_FILE):
        try:
            leads = json.load(open(LEADS_FILE, encoding="utf-8"))
        except Exception:
            leads = []
    leads.append({
        "name": name,
        "service": service,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    json.dump(leads, open(LEADS_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return len(leads)


def _save_lead_to_db(name: str, service: str) -> int:
    import psycopg2
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS leads ("
                "id SERIAL PRIMARY KEY, name TEXT, service TEXT, "
                "created_at TIMESTAMP DEFAULT NOW())"
            )
            cur.execute("INSERT INTO leads (name, service) VALUES (%s, %s)", (name, service))
            cur.execute("SELECT COUNT(*) FROM leads")
            total = cur.fetchone()[0]
        return total
    finally:
        conn.close()


def record_lead(name: str, service: str) -> int:
    if DATABASE_URL:
        total = _save_lead_to_db(name, service)
        where = "Postgres"
    else:
        total = _save_lead_to_file(name, service)
        where = "leads.json"
    print(f"📒 [CRM] Новая заявка: {name} — {service} (хранилище: {where}; всего: {total})")
    return total


# ── STRUCTURED OUTPUT — имя+задача в JSON ────────────────────────────
def extract_booking(text: str) -> dict:
    raw = ask(
        'Извлеки из сообщения имя клиента и что он хочет заказать (услугу/задачу). '
        'Верни СТРОГО JSON вида {"name": "...", "service": "..."}. '
        'Если чего-то нет — поставь null. '
        f"Только JSON.\n\nСообщение: {text}"
    )
    start, end = raw.find("{"), raw.rfind("}")
    try:
        return json.loads(raw[start:end + 1])
    except Exception:
        return {}


# ── JUDGE (опционально) ──────────────────────────────────────────────
def judge(question: str, answer: str) -> bool:
    verdict = ask(
        "Ответ вежливый и по делу? Первой строкой строго ДА или НЕТ.\n"
        f"Вопрос: {question}\nОтвет: {answer}"
    )
    first = verdict.strip().upper()
    return first.startswith("ДА") or first.startswith("YES")


SYSTEM_BASE = (
    "Ты — вежливый AI-консультант студии цифровых продуктов ELLHOME. "
    "ВАЖНОЕ ПРАВИЛО: всегда отвечай СТРОГО на языке последнего сообщения клиента. "
    "Английский вопрос — английский ответ. Русский вопрос — русский ответ. "
    "Отвечай ТОЛЬКО по фактам ниже; если факта нет — честно скажи и предложи "
    "оставить заявку. Будь краток, дружелюбен и по делу.\n\nФАКТЫ:\n" + KNOWLEDGE
)


def run_chat(message: str, history: list[dict]) -> dict:
    """RU: одно сообщение → ответ. Вся логика бота (Router→RAG→Tool→Judge)."""
    category = classify(message)

    system = SYSTEM_BASE
    if category == "цена":
        system += "\n\nВопрос про стоимость/сроки. Назови ориентир из фактов и уточни, что точная смета — после короткого брифа."
    if category == "заявка":
        system += "\n\nКлиент хочет оставить заявку. Если не хватает имени или описания задачи — вежливо уточни."

    # Memory — история приходит от виджета
    hist = [{"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in history]
    messages = [{"role": "system", "content": system}] + hist + [{"role": "user", "content": message}]
    answer = call_model(messages)

    lead = None
    if category == "заявка":
        data = extract_booking(message)
        if data.get("name") and data.get("service"):
            record_lead(data["name"], data["service"])
            lead = data
            answer += f"\n\n✅ Готово! Записал заявку: {data['name']} — {data['service']}. Скоро свяжусь: Telegram @M_B_lab."

    if USE_JUDGE and not judge(message, answer):
        answer = ask(f"Перепиши вежливее и по делу:\n{answer}", system=system)

    return {"reply": answer, "category": category, "lead": lead}


# ── HTTP API ─────────────────────────────────────────────────────────
app = FastAPI(title="ELLHOME bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class Msg(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    message: str
    history: list[Msg] = []


@app.get("/")
def health():
    return {"status": "ok", "provider": PROVIDER}


@app.post("/chat")
def chat_endpoint(body: ChatIn):
    history = [m.model_dump() for m in body.history]
    return run_chat(body.message, history)
