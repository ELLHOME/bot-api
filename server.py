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
import re
import random
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
CATEGORIES = ("цена", "заявка", "название", "идея", "гадание", "общее")
# какая категория к какому режиму относится ("общее" — остаётся в текущем)
CATEGORY_MODE = {"цена": "consult", "заявка": "consult",
                 "название": "lab", "идея": "lab",
                 "гадание": "guide"}


def classify(message: str) -> str:
    cat = ask(
        "Определи тип сообщения ОДНИМ словом из списка: цена, заявка, название, идея, гадание, общее.\n"
        "«цена» — сколько стоит, сроки, смета.\n"
        "«заявка» — хочет заказать, оставить контакт, начать проект.\n"
        "«название» — просит придумать имя, нейм, слоган для проекта/бренда/продукта.\n"
        "«идея» — просит придумать идею продукта, фичи, концепцию.\n"
        "«гадание» — просит погадать, предсказать, «книга/энциклопедия, ответь», спрашивает о судьбе.\n"
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




# ── ИНСТРУМЕНТЫ «ПУТЕВОДИТЕЛЯ» — открытый API Википедии ──────────────
# Домен зафиксирован, запросы только на чтение, с таймаутом и лимитом размера.
WIKI_API = "https://ru.wikipedia.org"
WIKI_TIMEOUT = 8
WIKI_UA = "ELLHOME-bot/1.0 (https://ellhome.github.io/Mikhail.Borgoyakov)"


def _wiki_get(url: str) -> dict:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": WIKI_UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=WIKI_TIMEOUT) as r:
        return json.loads(r.read(300_000).decode("utf-8"))


def _wiki_summary(payload: dict) -> dict:
    return {
        "title": payload.get("title", ""),
        "extract": (payload.get("extract") or "")[:1200],
        "url": (payload.get("content_urls", {}).get("desktop", {}) or {}).get("page", ""),
    }


# «Толщина» энциклопедии: идентификаторы статей в ru.wikipedia выданы примерно
# до этого номера (замерено по свежим статьям). Отсюда и границы в интерфейсе:
# страница 1–11 500, строка 1–99 — при них координата всегда внутри заселённого
# пространства. Число сверяем раз в год, оно только растёт.
WIKI_MAX_ID = 11_600_000
WIKI_MAX_PAGE = 11_500
WIKI_MAX_LINE = 99
WIKI_SHELVES = 6      # сколько «полок» по 50 номеров просматриваем подряд


def wiki_by_numbers(page: int, line: int) -> dict:
    """
    Гадание: числа, названные ЧЕЛОВЕКОМ, — это координаты в энциклопедии.
    Превращаем их в идентификатор статьи. Одни и те же числа всегда дают
    одну и ту же статью — выбор делает человек, а не бот.
    """
    import urllib.parse
    page, line = abs(int(page)), abs(int(line))
    # строка тоже заметно сдвигает координату, иначе соседние строки дают одно и то же.
    # По модулю — чтобы «страница 999999» не упиралась в пустоту за краем
    # энциклопедии, а завернулась внутрь: тупика не должно быть ни при каком вводе.
    target = (page * 1000 + line * 97) % WIKI_MAX_ID or 1

    # Идентификаторы идут с пропусками, поэтому одним запросом берём «полку»
    # подряд идущих номеров и выбираем ближайший существующий к загаданному.
    for attempt in range(WIKI_SHELVES):
        base = (target + attempt * 50) % WIKI_MAX_ID or 1
        ids = "|".join(str(base + i) for i in range(50))
        url = (f"{WIKI_API}/w/api.php?action=query&pageids={urllib.parse.quote(ids)}"
               "&prop=extracts|info&exintro=1&explaintext=1&inprop=url&format=json")
        pages = _wiki_get(url).get("query", {}).get("pages", {})
        found = [
            pg for pg in pages.values()
            # ns == 0 — только статьи. Без этого на полку попадают обсуждения,
            # профили участников, шаблоны и категории, и гадание выпадает на них.
            if "missing" not in pg and pg.get("ns") == 0
            and (pg.get("extract") or "").strip()
        ]
        if found:
            # из найденной «полки» выбираем по числам человека — детерминированно
            found.sort(key=lambda pg: int(pg["pageid"]))
            best = found[(page + line) % len(found)]
            return {
                "coordinates": f"страница {page}, строка {line}",
                "title": best.get("title", ""),
                "extract": (best.get("extract") or "")[:1200],
                "url": best.get("fullurl", ""),
            }
    return {"error": "на этих координатах энциклопедия молчит — попроси назвать другие числа"}


TOOLS_GUIDE = [
    {"type": "function", "function": {
        "name": "wiki_by_numbers",
        "description": ("Гадание по энциклопедии. Вызывай ТОЛЬКО когда человек сам назвал "
                        "номер страницы и номер строки. Никогда не придумывай числа за него."),
        "parameters": {"type": "object", "properties": {
            "page": {"type": "integer", "description": "Номер страницы, названный человеком"},
            "line": {"type": "integer", "description": "Номер строки, названный человеком"},
        }, "required": ["page", "line"]},
    }},
]

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

    if name == "wiki_by_numbers":
        try:
            return wiki_by_numbers(int(args.get("page", 0)), int(args.get("line", 0)))
        except Exception as e:
            return {"error": f"энциклопедия не отвечает: {e}"}

    return {"error": f"неизвестный инструмент: {name}"}


MAX_STEPS = 4   # предохранитель от зацикливания
LAST_AGENT_ERROR = ""   # временно: чтобы увидеть причину сбоя в ответе API


def agent_loop(messages: list[dict], tools: list | None = None) -> tuple[str, list[str]]:
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

    tools = tools if tools is not None else TOOLS
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
                tools=tools, tool_choice="auto",
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
    "ГРАНИЦА ТОНА: дерзко — не значит грубо. Запрещены мат и его эвфемизмы, "
    "туалетный юмор, шутки про смерть и болезни, а также любые подколы по национальности, "
    "внешности, возрасту, полу или достатку — включая слова «цыганщина», «колхоз», «быдло» "
    "в значении безвкусицы. Острота всегда про суть задачи, а не про людей. "
    "Это не смягчение: точный выпад по делу бьёт сильнее грубости.\n\n"
    "КАК НАДО: сначала про себя выпиши из запроса конкретные существительные, глаголы "
    "и жаргон этой области — и строй только из них. Общие слова о «сфере» и «нише» "
    "не годятся, годится материал самого запроса. "
    "Играй смыслами, звучанием и идиомами языка. Точная шутка лучше громкой. "
    "Одно меткое попадание ценнее пяти вежливых вариантов."
)

# ── Приёмы, из которых собираются варианты. На каждый запрос берутся случайные три:
# однообразие формы убивает ощущение выдумки сильнее любых штампов.
NAME_ANGLES = [
    "слово из чужой профессии, где оно значит совсем другое",
    "антоним того, чего ждёшь от такой вывески",
    "физический предмет, который есть в этом деле и звучит крепко",
    "обманка: выглядит как аббревиатура, а расшифровка бытовая",
    "устаревшее или диалектное слово, которое вдруг точно легло",
    "звук или шум, который издаёт само дело",
    "глагол в повелительном наклонении, короткий и нахальный",
    "профессиональный жаргонизм, понятный только своим",
    "часть идиомы, обрубленная на самом интересном месте",
    "имя собственное — прозвище, а не фамилия основателя",
    "единица измерения или величина из этой области",
    "слово с намеренной опечаткой, которая меняет смысл",
    "то, что клиент говорит вслух, когда всё наконец заработало",
    "название побочного эффекта, а не самой услуги",
]

IDEA_ANGLES = [
    "сделать публичным то, что все прячут",
    "отдать бесплатно самое дорогое, чтобы продать дешёвое",
    "перевернуть порядок шагов задом наперёд",
    "занять сторону того, кого в этой сфере принято ругать",
    "превратить недостаток в единственную примету",
    "обратиться не к клиенту, а к тому, кто на него влияет",
    "поставить срок или счётчик там, где их обычно нет",
    "сделать вручную то, что все автоматизировали, и наоборот",
    "вынести кухню наружу: показать процесс вместо результата",
    "придумать ритуал, который повторяют без вас",
]


def _angles(pool: list[str], n: int) -> str:
    return "\n".join(f"— {a}" for a in random.sample(pool, min(n, len(pool))))


LAB_DRAFT_NAMES = (
    "\n\nСЕЙЧАС ТЫ ЧЕРНОВИК. Накидай РОВНО 8 вариантов названия, каждый с новой строки, "
    "без нумерации, без объяснений, без форматирования — только сами названия.\n"
    "Три из них построй по этим приёмам, по одному на приём:\n{angles}\n"
    "Остальные пять — как придумается, но все разные по принципу. "
    "Плохие варианты тоже пиши: отбор будет потом."
)

LAB_PICK_NAMES = (
    "\n\nПеред тобой запрос и черновой список названий. Выбери ТРИ самых метких — "
    "и обязательно разных по принципу, а не три вариации одного. "
    "Слабые выброси без сожаления. Можешь слегка доточить выбранное.\n"
    "Формат ответа строго такой, без вступления:\n"
    "**Название**\nодна строка — почему оно попадает, с характером.\n\n"
    "(и так три раза, между блоками пустая строка)"
)

LAB_DRAFT_IDEA = (
    "\n\nСЕЙЧАС ТЫ ЧЕРНОВИК. Накидай РОВНО 5 разных идей, каждая одной строкой, "
    "без объяснений и форматирования.\n"
    "Две из них построй по этим приёмам:\n{angles}\n"
    "Идеи должны быть выполнимы руками, без бюджета и без фантастики."
)

LAB_PICK_IDEA = (
    "\n\nПеред тобой запрос и список черновых идей. Выбери ОДНУ самую неожиданную "
    "из тех, что реально выполнима, и разверни её. Остальные не упоминай.\n"
    "Формат строго такой, без вступления:\n"
    "**Что это:**\nодно предложение.\n\n"
    "**Неожиданный поворот:**\nодно-два предложения.\n\n"
    "**Почему сработает:**\nдве-три фразы, конкретно и с иронией."
)

# Черновик пишем горячо, отбор — холоднее: широкий разброс, затем трезвый выбор.
LAB_PICK_TEMP = float(os.getenv("LAB_PICK_TEMP", "0.8"))


def run_lab(message: str, hist: list[dict], category: str) -> str:
    """Два прохода: сначала много черновиков, потом отбор лучшего.
    Одного прохода мало — модель выдаёт первое пришедшее и на нём же успокаивается."""
    if category == "название":
        draft_sys = SYSTEM_LAB + LAB_DRAFT_NAMES.format(angles=_angles(NAME_ANGLES, 3))
        pick_sys = SYSTEM_LAB + LAB_PICK_NAMES
    elif category == "идея":
        draft_sys = SYSTEM_LAB + LAB_DRAFT_IDEA.format(angles=_angles(IDEA_ANGLES, 2))
        pick_sys = SYSTEM_LAB + LAB_PICK_IDEA
    else:
        answer = call_model(
            [{"role": "system", "content": SYSTEM_LAB + LAB_TASK["общее"]}] + hist
            + [{"role": "user", "content": message}], temperature=LAB_TEMP)
        if LAB_POLISH and not answer.startswith("⚠️"):
            polished = ask(POLISH_PROMPT + answer, system=SYSTEM_LAB, temperature=LAB_TEMP)
            if polished and not polished.startswith("⚠️"):
                answer = polished
        return answer

    drafts = call_model(
        [{"role": "system", "content": draft_sys}] + hist
        + [{"role": "user", "content": message}], temperature=LAB_TEMP)
    if not drafts or drafts.startswith("⚠️"):
        return drafts or ""

    picked = call_model([
        {"role": "system", "content": pick_sys},
        {"role": "user", "content": f"ЗАПРОС: {message}\n\nЧЕРНОВИКИ:\n{drafts}"},
    ], temperature=LAB_PICK_TEMP)
    return picked if picked and not picked.startswith("⚠️") else drafts

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
    "Сохрани язык, структуру и ссылку на источник, если она есть. "
    "Верни ТОЛЬКО итоговый текст, без комментариев.\n\nЧерновик:\n"
)

LAB_TEMP = float(os.getenv("LAB_TEMP", "1.15"))   # выше температура — меньше шаблонов
LAB_POLISH = os.getenv("LAB_POLISH", "true").lower() == "true"



# ── РЕЖИМ 3: ПУТЕВОДИТЕЛЬ (энциклопедия обо всём) ────────────────────
SYSTEM_GUIDE = (
    "Ты — «wikiмантия»: невозмутимый энциклопедический справочник, которого заставили гадать.\n"
    "Ты НЕ гадалка, НЕ мистик, НЕ коуч и НЕ астролог. Интонация гороскопа строго запрещена: "
    "никаких «символизмов», «энергий», «знаков судьбы» и обещаний успеха. "
    "Ты сухо сообщаешь, что выпало, и так же сухо связываешь это с вопросом — "
    "с лицом справочника, которому эта затея кажется сомнительной, но правила есть правила.\n"
    "Отвечай на языке последнего сообщения собеседника.\n\n"
    "ГОЛОС: невозмутимый тон энциклопедии, которая абсолютно уверена в себе — "
    "сообщает факты с каменным лицом, а выводы делает житейские и абсурдные. "
    "Сухо и коротко. Юмор рождается из контраста серьёзной подачи и нелепого итога. "
    "Никогда не объясняй шутку и не подмигивай читателю.\n\n"
    "ЧЕСТНОСТЬ: факты, даты и числа бери ТОЛЬКО из результата инструмента. "
    "Выдумывать можно исключительно комментарии и выводы, но не сами сведения.\n\n"
    "ФОРМАТ РЕЗУЛЬТАТА — применяется ТОЛЬКО когда ты выдаёшь толкование:\n"
    "**Название** — одно-два предложения сути по фактам источника.\n"
    "Затем 2–3 фразы фирменного комментария: неожиданный угол, бытовая аналогия, вывод.\n"
    "Последняя строка курсивом: «Степень опасности: … Рекомендация: …» — абсурдная, но в тему.\n"
    "В самом конце — ССЫЛКА из поля url результата инструмента, в неизменном виде "
    "(полный адрес). Не заменяй её словами вроде «Источник: Энциклопедия, стр. 137» "
    "и не выдумывай адрес сам.\n\n"
    "А ВОТ ЕСЛИ ты задаёшь уточняющий вопрос или ведёшь человека по шагам обряда — "
    "отвечай обычной короткой фразой в одно-два предложения: без заголовка, "
    "без «Степени опасности», без ссылок. Формат статьи тут неуместен.\n\n"
    "ЗАПРЕЩЕНО: канцелярит, «в современном мире», восторги, гирлянды эмодзи, "
    "пересказ статьи целиком, длинные списки, извинения за краткость."
)

GUIDE_TASK = {
    "гадание": (
        "\n\nЧеловек гадает. РИТУАЛ СВЯЩЕНЕН: числа называет ОН САМ, ты их не выбираешь "
        "и не предлагаешь. Обряд идёт строго в два шага:\n"
        "ШАГ 1. Вопроса ещё нет — попроси написать вопрос. Ничего не вызывай.\n"
        "ШАГ 2. Вопрос есть, чисел нет — одной короткой фразой подтверди, что вопрос принят, "
        f"и попроси назвать страницу (1–{WIKI_MAX_PAGE}) и строку (1–{WIKI_MAX_LINE}). "
        "Без формата статьи. Ничего не вызывай.\n"
        "ШАГ 3. Есть и вопрос, и числа — гадай.\n"
        "Если человек прислал всё сразу — не заставляй повторять, сразу переходи к шагу 3.\n"
        "Когда числа названы — вызови wiki_by_numbers и истолкуй найденное СТРОГО так:\n"
        "1) одна строка: какие координаты и что на них обнаружено;\n"
        "2) одно предложение дословно из статьи, в кавычках;\n"
        "3) толкование применительно к вопросу — 2–3 сухие фразы, чем неожиданнее связь, тем лучше;\n"
        "4) последняя строка курсивом: «Степень опасности: … Рекомендация: …».\n"
        "ЗАПРЕЩЕНО: заголовки, нумерованные списки, разбор по пунктам, вступления, "
        "мотивационные концовки («решать только вам», «главное — верить», «удачи!»), "
        "слова «символизм», «трансформация», «энергия», «вселенная», «неслучайно». "
        "Толкование должно быть остроумным и сухим, а не возвышенным. Коротко."
    ),
}

# Толкование пишет модель, но координаты и ссылку подставляет код —
# так они не теряются и не выдумываются.
GUIDE_INTERPRET = (
    "\n\nСтатья УЖЕ выпала — искать и выбирать ничего не нужно. "
    "Координаты и ссылку на источник подставит система, твоя задача — ТОЛЬКО толкование.\n"
    "Пиши ровно так, без заголовков и без списков:\n"
    "Абзац 1 — одно-два предложения о том, что это такое, строго по фактам источника.\n"
    "Абзац 2 — 2–3 сухие фразы: как это отвечает на вопрос человека. "
    "Чем неожиданнее связь, тем лучше.\n"
    "Абзац 3 — одна строка курсивом между звёздочками: "
    "*Степень опасности: … Рекомендация: …* — абсурдная, но в тему.\n"
    "НЕ повторяй координаты, НЕ пиши название статьи отдельной строкой, "
    "НЕ вставляй ссылок и адресов — всё это добавит система.\n"
    "ЗАПРЕЩЕНО: вступления, нумерованные списки, мотивационные концовки, "
    "слова «символизм», «трансформация», «энергия», «вселенная», «неслучайно»."
)


def _strip_guide_noise(text: str) -> str:
    """Убираем то, что модель дублирует: шапку с названием, ссылки, адреса."""
    t = (text or "").strip()
    t = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", t)   # markdown-ссылки
    t = re.sub(r"https?://\S+", "", t)                          # голые адреса
    lines = [l for l in t.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    # первая строка вида «**Что-то**» или «страница 12, строка 3 → …» — это наша шапка
    if lines:
        head = lines[0].strip()
        if re.fullmatch(r"\*\*[^*]{1,80}\*\*[.:]?", head) or re.match(
            r"^(страница|page)\s*\d+", head, re.I
        ):
            lines.pop(0)
    return "\n".join(lines).strip()


def _guide_question(hist: list[dict], message: str) -> str:
    """Вопрос человека — последняя его реплика, где есть слова, а не только числа."""
    for m in reversed(hist):
        if m.get("role") != "user":
            continue
        text = str(m.get("content", "")).strip()
        if len(re.sub(r"[\d\s.,;:!?-]", "", text)) >= 3:
            return text
    return message


def run_chat(message: str, history: list[dict], mode: str = "consult") -> dict:
    """Router выбирает режим. Деловой режим — агент с инструментами, Лаборатория — творчество."""
    category = classify(message)
    requested = mode if mode in ("consult", "lab", "guide") else "consult"
    mode = CATEGORY_MODE.get(category, requested)

    # На вкладке гадания любой обычный вопрос — это вопрос к оракулу, а не болтовня.
    # Без этого «стоит ли менять работу?» уезжало в «общее», и ритуал не начинался.
    if requested == "guide" and mode == "guide":
        category = "гадание"

    hist = [{"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in history]
    tools_used: list[str] = []

    # Шаги обряда ведём кодом, а не промптом: пока не названы два числа,
    # отвечаем короткой фразой сами — модель тут только мешает (любит формат статьи).
    if mode == "guide" and category == "гадание" and len(re.findall(r"\d+", message)) < 2:
        ru = any("\u0400" <= ch <= "\u04ff" for ch in message)
        stop = {"погадай", "погадать", "гадание", "гадай", "divine", "divination"}
        words = [w for w in re.sub(r"[^\w\s]", " ", message.lower()).split() if w not in stop]
        if len(words) < 2:
            reply = (f"Напишите свой вопрос — а следом назовите страницу "
                     f"(1–{WIKI_MAX_PAGE:,}) и строку (1–{WIKI_MAX_LINE})."
                     .replace(",", " ")
                     if ru else
                     f"Write your question — then name a page (1–{WIKI_MAX_PAGE:,}) "
                     f"and a line (1–{WIKI_MAX_LINE}).")
        else:
            reply = (f"Вопрос принят. Теперь назовите страницу "
                     f"(1–{WIKI_MAX_PAGE:,}) и строку (1–{WIKI_MAX_LINE})."
                     .replace(",", " ")
                     if ru else
                     f"Question received. Now name a page (1–{WIKI_MAX_PAGE:,}) "
                     f"and a line (1–{WIKI_MAX_LINE}).")
        return {"reply": reply, "category": category, "mode": mode, "lead": None, "tools": []}

    # Гадание с числами ведём кодом: сами ходим в энциклопедию, сами ставим
    # шапку с координатами и ссылку. Модели остаётся только толкование —
    # иначе она то теряет координаты, то придумывает адрес источника.
    if mode == "guide" and category == "гадание":
        nums = [int(n) for n in re.findall(r"\d+", message)[:2]]
        ru = any("\u0400" <= ch <= "\u04ff" for ch in message + " ".join(
            str(m.get("content", "")) for m in hist))
        try:
            found = wiki_by_numbers(nums[0], nums[1])
        except Exception as e:                      # энциклопедия недоступна
            found = {"error": str(e)}
        tools_used = ["wiki_by_numbers"]
        if found.get("error") or not found.get("title"):
            reply = ("На этих координатах энциклопедия молчит. Назовите другие числа."
                     if ru else
                     "The encyclopedia is silent at those coordinates. Name other numbers.")
            return {"reply": reply, "category": category, "mode": mode,
                    "lead": None, "tools": tools_used}

        question = _guide_question(hist, message)
        system = SYSTEM_GUIDE + GUIDE_INTERPRET
        ctx = (f"ВОПРОС ЧЕЛОВЕКА: {question}\n\n"
               f"ЧТО ВЫПАЛО: {found['title']}\n"
               f"ФАКТЫ ИЗ ИСТОЧНИКА (только они, ничего не додумывай):\n{found['extract']}")
        body = _strip_guide_noise(call_model(
            [{"role": "system", "content": system}, {"role": "user", "content": ctx}],
            temperature=LAB_TEMP,
        ))
        if not body or body.startswith("⚠️"):
            body = (found["extract"] or "").strip()[:400]

        coords = (f"страница {nums[0]}, строка {nums[1]}" if ru
                  else f"page {nums[0]}, line {nums[1]}")
        answer = f"**{coords} → {found['title']}**\n\n{body}"
        if found.get("url"):
            label = "Статья целиком" if ru else "Full article"
            answer += f"\n\n[{label}]({found['url']})"
        return {"reply": answer, "category": category, "mode": mode,
                "lead": None, "tools": tools_used}

    if mode == "guide":
        system = SYSTEM_GUIDE + GUIDE_TASK.get(category, GUIDE_TASK["гадание"])
        messages = [{"role": "system", "content": system}] + hist + [{"role": "user", "content": message}]
        answer, tools_used = agent_loop(messages, TOOLS_GUIDE)
        if LAB_POLISH and not answer.startswith("⚠️"):
            polished = ask(POLISH_PROMPT + answer, system=SYSTEM_GUIDE, temperature=LAB_TEMP)
            if polished and not polished.startswith("⚠️"):
                answer = polished
    elif mode == "lab":
        answer = run_lab(message, hist, category)
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
    mode: str = "consult"   # "consult" | "lab" | "guide"


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
