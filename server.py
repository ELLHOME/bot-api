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

# Структурированный прайс — единственный источник правды по ценам (инструмент get_price)
SERVICES = json.load(open(os.path.join(os.path.dirname(__file__), "services.json"), encoding="utf-8"))

LEADS_FILE = os.path.join(os.path.dirname(__file__), "leads.json")
DATABASE_URL = os.getenv("DATABASE_URL")  # задаётся хостингом при подключении Postgres


# ── Вызов модели / Model call ────────────────────────────────────────
def _gemini_client():
    """Клиент Gemini через OpenAI-совместимый endpoint (нужен и обычному вызову, и агенту)."""
    from openai import OpenAI
    key = os.getenv("GOOGLE_API_KEY")
    if not key or "..." in key:
        return None
    return OpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


def call_model(messages: list[dict], temperature: float | None = None) -> str:
    try:
        if PROVIDER == "ollama":
            import ollama  # RU: ленивый импорт — на сервере ollama не нужен
            opts = {"temperature": temperature} if temperature is not None else None
            return (ollama.chat(model=OLLAMA_MODEL, messages=messages, options=opts).message.content or "")

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
        return (client.chat.completions.create(
            model=GEMINI_MODEL, messages=messages, **kw
        ).choices[0].message.content or "")
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
def _save_lead_to_file(name: str, service: str, contact: str = "") -> int:
    leads = []
    if os.path.exists(LEADS_FILE):
        try:
            leads = json.load(open(LEADS_FILE, encoding="utf-8"))
        except Exception:
            leads = []
    leads.append({
        "name": name,
        "service": service,
        "contact": contact,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    json.dump(leads, open(LEADS_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return len(leads)


def _save_lead_to_db(name: str, service: str, contact: str = "") -> int:
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
            # безопасная миграция: добавляем колонку контакта, если её ещё нет
            cur.execute("ALTER TABLE leads ADD COLUMN IF NOT EXISTS contact TEXT")
            cur.execute(
                "INSERT INTO leads (name, service, contact) VALUES (%s, %s, %s)",
                (name, service, contact),
            )
            cur.execute("SELECT COUNT(*) FROM leads")
            total = cur.fetchone()[0]
        return total
    finally:
        conn.close()


def record_lead(name: str, service: str, contact: str = "") -> int:
    # RU: сначала пробуем базу (если задана); при любой ошибке — не роняем чат,
    #     а откатываемся на файл. EN: try DB first, fall back to file on any error.
    if DATABASE_URL:
        try:
            total = _save_lead_to_db(name, service, contact)
            print(f"📒 [CRM] Новая заявка → Postgres: {name} — {service} (всего: {total})")
            return total
        except Exception as e:
            print(f"⚠️ Postgres недоступен ({e}); пишу заявку в файл.")
    total = _save_lead_to_file(name, service, contact)
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



# ── ИНСТРУМЕНТЫ АГЕНТА (белый список — никакого шелла и файлов) ──────
TOOLS = [
    {"type": "function", "function": {
        "name": "list_services",
        "description": "Полный список услуг ELLHOME с ценами «от» и сроками. Вызывай, когда спрашивают, что вы умеете, или просят прайс целиком.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_price",
        "description": "Точная цена и срок по одной услуге. ВСЕГДА вызывай перед тем, как назвать цену — не придумывай цифры сам.",
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string", "description": "Ключ или название услуги: landing, corporate, shop, bot, webapp, 3d, motion, branding, consult"}
        }, "required": ["service"]},
    }},
    {"type": "function", "function": {
        "name": "save_lead",
        "description": "Записать заявку клиента в CRM. Вызывай, когда известны имя и суть задачи. Контакт (телеграм/почта/телефон) передавай, если человек его назвал.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Имя клиента"},
            "task": {"type": "string", "description": "Что нужно сделать"},
            "contact": {"type": "string", "description": "Телеграм, почта или телефон, если назван"},
        }, "required": ["name", "task"]},
    }},
]


def execute_tool(name: str, args: dict) -> dict:
    """Исполняем инструмент. Всё строго ограничено — только эти три действия."""
    if name == "list_services":
        return {
            "currency": SERVICES.get("currency"),
            "terms": SERVICES.get("terms_ru"),
            "services": [
                {"key": x["key"], "name": x["ru"], "from": x["from"], "term": x["term_ru"]}
                for x in SERVICES["services"]
            ],
        }

    if name == "get_price":
        q = str(args.get("service", "")).strip().lower()
        for x in SERVICES["services"]:
            if q and (q == x["key"] or q in x["ru"].lower() or q in x["en"].lower()):
                return {"name": x["ru"], "from": x["from"], "currency": SERVICES["currency"],
                        "term": x["term_ru"], "terms": SERVICES["terms_ru"]}
        return {"error": "услуга не найдена",
                "available": [x["key"] for x in SERVICES["services"]]}

    if name == "save_lead":
        nm = str(args.get("name", "")).strip()[:120]
        task = str(args.get("task", "")).strip()[:400]
        contact = str(args.get("contact", "")).strip()[:200]
        if not nm or not task:
            return {"ok": False, "error": "нужны имя и описание задачи"}
        total = record_lead(nm, task, contact)
        return {"ok": True, "saved": {"name": nm, "task": task, "contact": contact}, "total": total}

    return {"error": f"неизвестный инструмент: {name}"}


MAX_STEPS = 4   # предохранитель от зацикливания
LAST_AGENT_ERROR = ""   # временно: чтобы увидеть причину сбоя в ответе API


def agent_loop(messages: list[dict]) -> tuple[str, list[str]]:
    """
    Агентский цикл. Модель сама решает, какие инструменты дёрнуть; сервер исполняет
    и подмешивает результаты в контекст следующего захода.

    Почему не «классические» tool-сообщения: Gemini 3 — думающая модель и требует
    возвращать вместе с вызовом её thought_signature, которая через OpenAI-совместимый
    слой не отдаётся. Поэтому результаты передаём контекстом — работает на любой модели.
    """
    global LAST_AGENT_ERROR
    LAST_AGENT_ERROR = ""
    if PROVIDER != "gemini":
        return call_model(messages), []          # локально на ollama — без инструментов
    client = _gemini_client()
    if client is None:
        return call_model(messages), []

    used: list[str] = []
    results: list[str] = []
    done: dict[str, str] = {}                    # кэш: один и тот же вызов не повторяем

    def with_results(base: list[dict]) -> list[dict]:
        if not results:
            return list(base)
        return list(base) + [{
            "role": "system",
            "content": ("РЕЗУЛЬТАТЫ УЖЕ ВЫПОЛНЕННЫХ ИНСТРУМЕНТОВ. Повторно их не вызывай — "
                        "отвечай клиенту на основе этих данных:\n" + "\n".join(results)),
        }]

    try:
        for _ in range(MAX_STEPS):
            resp = client.chat.completions.create(
                model=GEMINI_MODEL, messages=with_results(messages),
                tools=TOOLS, tool_choice="auto",
            )
            m = resp.choices[0].message
            calls = getattr(m, "tool_calls", None)
            if not calls:
                text = (m.content or "").strip()
                if text:
                    return text, used

            fresh = False
            for c in (calls or []):
                name = c.function.name
                try:
                    args = json.loads(c.function.arguments or "{}")
                except Exception:
                    args = {}
                key = f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
                if key in done:
                    continue
                res = execute_tool(name, args)
                done[key] = "1"
                used.append(name)
                fresh = True
                print(f"🔧 [tool] {name}({args}) -> {res}")
                results.append(f"{name}({json.dumps(args, ensure_ascii=False)}) -> "
                               f"{json.dumps(res, ensure_ascii=False)}")
            if not fresh:
                break                             # новых вызовов нет — идём за финальным ответом

        # финальный заход БЕЗ инструментов: гарантированно получаем текст
        final = client.chat.completions.create(
            model=GEMINI_MODEL, messages=with_results(messages)
        )
        return (final.choices[0].message.content or ""), used

    except Exception as e:
        import traceback
        LAST_AGENT_ERROR = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        fallback = call_model(messages)
        if not (fallback or "").strip():
            fallback = ("Сейчас не могу свериться с прайсом — напишите, пожалуйста, "
                        "в Telegram @M_B_lab, отвечу лично.")
        return fallback, used


# ── РЕЖИМ 1: КОНСУЛЬТАНТ (по базе знаний) ────────────────────────────
SYSTEM_CONSULT = (
    "Ты — вежливый AI-консультант студии цифровых продуктов ELLHOME. "
    "ВАЖНОЕ ПРАВИЛО: всегда отвечай СТРОГО на языке последнего сообщения клиента. "
    "Английский вопрос — английский ответ. Русский вопрос — русский ответ. "
    "Отвечай ТОЛЬКО по фактам ниже; если факта нет — честно скажи и предложи "
    "оставить заявку. Будь краток, дружелюбен и по делу.\n\n"
    "У ТЕБЯ ЕСТЬ ИНСТРУМЕНТЫ, пользуйся ими:\n"
    "— НИКОГДА не называй цену или срок по памяти: сначала вызови get_price "
    "(или list_services, если просят прайс целиком) и отвечай по его результату.\n"
    "— Когда клиент готов оставить заявку и назвал имя и суть задачи — вызови save_lead. "
    "Если контакт не назван, сначала вежливо попроси телеграм или почту, потом сохраняй.\n"
    "— Не выдумывай услуг, которых нет в прайсе.\n\nФАКТЫ:\n" + KNOWLEDGE
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
    """Router выбирает режим. Деловой режим — агент с инструментами, Лаборатория — творчество."""
    category = classify(message)
    mode = CATEGORY_MODE.get(category, mode if mode in ("consult", "lab") else "consult")

    hist = [{"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in history]
    tools_used: list[str] = []

    if mode == "lab":
        system = SYSTEM_LAB + LAB_TASK.get(category, LAB_TASK["общее"])
        messages = [{"role": "system", "content": system}] + hist + [{"role": "user", "content": message}]
        answer = call_model(messages, temperature=LAB_TEMP)
        if LAB_POLISH and not answer.startswith("⚠️"):
            polished = ask(POLISH_PROMPT + answer, system=SYSTEM_LAB, temperature=LAB_TEMP)
            if polished and not polished.startswith("⚠️"):
                answer = polished
    else:
        system = SYSTEM_CONSULT
        messages = [{"role": "system", "content": system}] + hist + [{"role": "user", "content": message}]
        answer, tools_used = agent_loop(messages)     # модель сама решает про инструменты
        if USE_JUDGE and not judge(message, answer):
            answer = ask(f"Перепиши вежливее и по делу:\n{answer}", system=system)

    if not (answer or "").strip():
        answer = ("Секунду, что-то пошло не так с ответом. Попробуйте переспросить "
                  "или напишите в Telegram @M_B_lab.")

    return {
        "reply": answer,
        "debug": LAST_AGENT_ERROR or None,   # временно, для отладки
        "category": category,
        "mode": mode,
        "lead": {"saved": True} if "save_lead" in tools_used else None,
        "tools": tools_used,
    }


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
