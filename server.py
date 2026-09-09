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
import time
import datetime

from fastapi import FastAPI, Request
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

# ── Лимиты (защита от спама и от лишних трат) ────────────────────────
RATE_WINDOW = int(os.getenv("RATE_WINDOW", "600"))   # окно, сек (10 мин)
RATE_MAX = int(os.getenv("RATE_MAX", "15"))          # сообщений за окно с одного IP
MAX_MESSAGE_LEN = 1000                                # максимум символов в сообщении
MAX_HISTORY = 20                                      # сколько последних реплик шлём модели
_hits: dict[str, list[float]] = {}


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")   # за прокси Render
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_ok(ip: str) -> bool:
    now = time.time()
    hits = [t for t in _hits.get(ip, []) if now - t < RATE_WINDOW]
    if len(hits) >= RATE_MAX:
        _hits[ip] = hits
        return False
    hits.append(now)
    _hits[ip] = hits
    if len(_hits) > 500:   # лёгкая уборка старых записей
        for k in [k for k, v in _hits.items() if not any(now - t < RATE_WINDOW for t in v)]:
            _hits.pop(k, None)
    return True

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
def call_model(messages: list[dict], temperature: float | None = None) -> str:
    try:
        if PROVIDER == "ollama":
            import ollama  # RU: ленивый импорт — на сервере ollama не нужен
            opts = {"temperature": temperature} if temperature is not None else None
            return ollama.chat(model=OLLAMA_MODEL, messages=messages, options=opts).message.content

        # cloud — Gemini через OpenAI-совместимый endpoint
        from openai import OpenAI
        key = os.getenv("GOOGLE_API_KEY")
        if not key or "..." in key:
            return "⚠️ На сервере не задан GOOGLE_API_KEY (нужен для PROVIDER='gemini')."
        client = OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        kw = {"temperature": temperature} if temperature is not None else {}
        return client.chat.completions.create(
            model=GEMINI_MODEL, messages=messages, **kw
        ).choices[0].message.content
    except Exception as e:
        return f"⚠️ Модель не отвечает (PROVIDER={PROVIDER}): {e}"


def ask(prompt: str, system: str = "Ты — помощник.", temperature: float | None = None) -> str:
    return call_model([
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ], temperature=temperature)


# ── ROUTER — один вызов определяет и тему, и режим ───────────────────
CATEGORIES = ("цена", "заявка", "название", "идея", "общее")
# какая категория к какому режиму относится ("общее" — остаётся в текущем)
CATEGORY_MODE = {"цена": "consult", "заявка": "consult", "название": "lab", "идея": "lab"}


def classify(message: str) -> str:
    cat = ask(
        "Определи тип сообщения ОДНИМ словом из списка: цена, заявка, название, идея, общее.\n"
        "«цена» — сколько стоит, сроки, смета.\n"
        "«заявка» — хочет заказать, оставить контакт, начать проект.\n"
        "«название» — просит придумать имя, нейм, слоган для проекта/бренда/продукта.\n"
        "«идея» — просит придумать идею продукта, фичи, концепцию.\n"
        "«общее» — всё остальное.\n"
        f"Верни только слово.\n\nСообщение: {message}"
    ).strip().lower()
    for key in CATEGORIES:
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
    # connect_timeout — чтобы недоступная база падала быстро (5с), а не висела минутами
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
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
    # RU: сначала пробуем базу (если задана); при любой ошибке — не роняем чат,
    #     а откатываемся на файл. EN: try DB first, fall back to file on any error.
    if DATABASE_URL:
        try:
            total = _save_lead_to_db(name, service)
            print(f"📒 [CRM] Новая заявка → Postgres: {name} — {service} (всего: {total})")
            return total
        except Exception as e:
            print(f"⚠️ Postgres недоступен ({e}); пишу заявку в файл.")
    total = _save_lead_to_file(name, service)
    print(f"📒 [CRM] Новая заявка → leads.json: {name} — {service} (всего: {total})")
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


# ── РЕЖИМ 1: КОНСУЛЬТАНТ (по базе знаний) ────────────────────────────
SYSTEM_CONSULT = (
    "Ты — вежливый AI-консультант студии цифровых продуктов ELLHOME. "
    "ВАЖНОЕ ПРАВИЛО: всегда отвечай СТРОГО на языке последнего сообщения клиента. "
    "Английский вопрос — английский ответ. Русский вопрос — русский ответ. "
    "Отвечай ТОЛЬКО по фактам ниже; если факта нет — честно скажи и предложи "
    "оставить заявку. Будь краток, дружелюбен и по делу.\n\nФАКТЫ:\n" + KNOWLEDGE
)

# ── РЕЖИМ 2: ЛАБОРАТОРИЯ (идеи и нейминг) ────────────────────────────
# Главная задача промпта — выбить из модели её штампы. Поэтому запреты
# перечислены буквально: модель хорошо избегает того, что названо явно.
SYSTEM_LAB = (
    "Ты — «Лаборатория ELLHOME»: остроумный напарник по идеям и названиям.\n"
    "Всегда отвечай на языке последнего сообщения собеседника.\n\n"
    "ТОН: живой и уверенный, юмор взрослого человека, а не корпоративного буклета. "
    "Коротко. Без вступлений вроде «Отличный вопрос!» и без концовок вроде «Надеюсь, это поможет!».\n\n"
    "СТРОГО ЗАПРЕЩЕНО (нарушение = провальный ответ):\n"
    "— названия с кусками: -ify, -ly, -hub, -nova, -sphere, -genix, -mind, Tech, Smart, Neo, "
    "Digital, Кибер, Умный, Про, Мега, Супер;\n"
    "— слова-пустышки: Синергия, Импульс, Вектор, Горизонт, Прорыв, Экосистема, Инновация, Платформа;\n"
    "— слоганы вида «больше чем просто…», «нового поколения», «ваш надёжный партнёр», "
    "«решение, которое меняет всё»;\n"
    "— обороты «в современном мире», «в эпоху цифровизации», «динамично развивающийся»;\n"
    "— вежливая вода, восторги, гирлянды эмодзи, длинные списки банальностей.\n\n"
    "КАК НАДО: зацепись за конкретную деталь запроса и вытащи из неё неожиданный угол. "
    "Играй смыслами, звучанием и идиомами языка. Точная шутка лучше громкой. "
    "Одно меткое попадание ценнее пяти вежливых вариантов."
)

LAB_TASK = {
    "название": (
        "\n\nПросят НАЗВАНИЕ. Дай ровно 3 варианта, разных по характеру:\n"
        "1) короткое и хлёсткое;\n"
        "2) с каламбуром или двойным дном;\n"
        "3) дерзкое — которое сначала царапает, а потом нравится.\n"
        "После каждого — одна короткая строка «почему», тоже с характером. "
        "Никаких «плюсов и минусов» и рассуждений про целевую аудиторию."
    ),
    "идея": (
        "\n\nПросят ИДЕЮ. Дай ОДНУ идею, не список. Три коротких блока: "
        "что это (одно предложение), в чём неожиданный поворот, и почему сработает — "
        "конкретно и с иронией. Идея должна быть выполнима руками, а не фантастика."
    ),
    "общее": (
        "\n\nПоддержи разговор в том же духе: коротко, живо, по делу. "
        "Если спрашивают про заказ, цены или услуги — скажи, что это к «Консультанту» "
        "(соседняя вкладка), и предложи переключиться."
    ),
}

# Второй проход: вычищаем то, что всё-таки прозвучало по-нейросетевому
POLISH_PROMPT = (
    "Ниже черновик ответа. Перепиши его так, чтобы он звучал живо и небанально: "
    "выкинь всё похожее на типичный текст нейросети (шаблонные названия, канцелярит, "
    "вежливую воду, восторги, штампы), усиль самое меткое, сократи. "
    "Сохрани язык и структуру. Верни ТОЛЬКО итоговый текст, без комментариев.\n\nЧерновик:\n"
)

LAB_TEMP = float(os.getenv("LAB_TEMP", "1.15"))   # выше температура — меньше шаблонов
LAB_POLISH = os.getenv("LAB_POLISH", "true").lower() == "true"


def run_chat(message: str, history: list[dict], mode: str = "consult") -> dict:
    """Одно сообщение → ответ. Router определяет тему и при необходимости меняет режим."""
    category = classify(message)

    # Router может переключить режим: попросили название/идею — уходим в «Лабораторию»,
    # спросили про цену/заказ — возвращаемся к «Консультанту». «Общее» режим не меняет.
    mode = CATEGORY_MODE.get(category, mode if mode in ("consult", "lab") else "consult")

    if mode == "lab":
        system = SYSTEM_LAB + LAB_TASK.get(category, LAB_TASK["общее"])
        temperature = LAB_TEMP
    else:
        system = SYSTEM_CONSULT
        temperature = None
        if category == "цена":
            system += "\n\nВопрос про стоимость/сроки. Назови ориентир из фактов и уточни, что точная смета — после короткого брифа."
        if category == "заявка":
            system += "\n\nКлиент хочет оставить заявку. Если не хватает имени или описания задачи — вежливо уточни."

    # Memory — история приходит от виджета
    hist = [{"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in history]
    messages = [{"role": "system", "content": system}] + hist + [{"role": "user", "content": message}]
    answer = call_model(messages, temperature=temperature)

    # Tool Calling — заявки собираем только в режиме консультанта
    lead = None
    if mode == "consult" and category == "заявка":
        data = extract_booking(message)
        if data.get("name") and data.get("service"):
            record_lead(data["name"], data["service"])
            lead = data
            answer += f"\n\n✅ Готово! Записал заявку: {data['name']} — {data['service']}. Скоро свяжусь: Telegram @M_B_lab."

    # «Судья остроумия» — второй проход только для Лаборатории
    if mode == "lab" and LAB_POLISH and not answer.startswith("⚠️"):
        polished = ask(POLISH_PROMPT + answer, system=SYSTEM_LAB, temperature=LAB_TEMP)
        if polished and not polished.startswith("⚠️"):
            answer = polished

    if mode == "consult" and USE_JUDGE and not judge(message, answer):
        answer = ask(f"Перепиши вежливее и по делу:\n{answer}", system=system)

    return {"reply": answer, "category": category, "lead": lead, "mode": mode}


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
    mode: str = "consult"   # "consult" | "lab"


@app.get("/")
def health():
    return {"status": "ok", "provider": PROVIDER}


@app.post("/chat")
def chat_endpoint(body: ChatIn, request: Request):
    message = (body.message or "").strip()[:MAX_MESSAGE_LEN]
    if not message:
        return {"reply": "", "category": "empty", "lead": None, "mode": body.mode}

    if not _rate_ok(_client_ip(request)):
        ru = any("\u0400" <= ch <= "\u04ff" for ch in message)
        return {
            "reply": ("Слишком много сообщений подряд — попробуйте, пожалуйста, через несколько минут. "
                      "Если вопрос срочный, напишите в Telegram @M_B_lab."
                      if ru else
                      "Too many messages in a row — please try again in a few minutes. "
                      "If it's urgent, message Telegram @M_B_lab."),
            "category": "limit",
            "lead": None,
            "mode": body.mode,
        }

    try:
        history = [m.model_dump() for m in body.history][-MAX_HISTORY:]
        return run_chat(message, history, body.mode)
    except Exception:
        import traceback
        traceback.print_exc()  # виден в логах Render
        return {
            "reply": "⚠️ Небольшая техническая заминка. Попробуйте ещё раз или напишите в Telegram @M_B_lab.",
            "category": "error",
            "lead": None,
            "mode": body.mode,
        }
