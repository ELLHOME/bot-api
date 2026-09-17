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
# события счётчика идут пачками (визит — это десяток строк), лимит свободнее
EVENT_RATE_MAX = int(os.getenv("EVENT_RATE_MAX", "120"))
MAX_MESSAGE_LEN = 1000                                # максимум символов в сообщении
MAX_HISTORY = 20                                      # сколько последних реплик шлём модели
_hits: dict[str, list[float]] = {}


# ── УВЕДОМЛЕНИЯ В ТЕЛЕГРАМ ───────────────────────────────────────────
# Заявка, о которой узнаёшь через два дня, — потерянная заявка. И молчащий
# бот хуже отсутствующего: посетитель решит, что сломан весь сайт.
# Поэтому оба события уходят в личку владельцу.
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()
LAST_NOTIFY = "ещё не отправляли"   # видно в / — что ответил телеграм
_notified: dict[str, float] = {}      # чтобы поломка не писала сто раз подряд


def notify_owner(text: str, throttle_key: str = "", throttle_sec: int = 3600) -> bool:
    """Шлёт сообщение владельцу. Никогда не роняет чат: не настроено или
    телеграм недоступен — просто возвращает False."""
    global LAST_NOTIFY
    if not TG_TOKEN or not TG_CHAT:
        LAST_NOTIFY = "не задан токен или chat_id"
        return False
    if throttle_key:
        last = _notified.get(throttle_key, 0)
        if time.time() - last < throttle_sec:
            return False
        _notified[throttle_key] = time.time()
    try:
        import urllib.request
        import urllib.parse
        data = urllib.parse.urlencode({
            "chat_id": TG_CHAT,
            "text": text[:3900],
            "disable_web_page_preview": "true",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data=data)
        with urllib.request.urlopen(req, timeout=8) as r:
            LAST_NOTIFY = f"ok {r.status}"
            return r.status == 200
    except Exception as e:
        LAST_NOTIFY = f"ошибка: {e}"[:200]
        print(f"⚠️ Телеграм не принял уведомление: {e}")
        return False


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")   # за прокси Render
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_ok(ip: str, limit: int | None = None, bucket: str = "") -> bool:
    """Счётчик обращений с одного адреса.

    bucket разводит разные вещи по разным счётчикам. Без него выходило так:
    страница при загрузке шлёт событие счётчика, потом подсказки городов на
    каждую букву, потом расчёт карты — и на вопрос к карте с её лимитом
    в восемь обращений места уже не остаётся. Считать их вместе нельзя:
    это разные действия с разной ценой.
    """
    now = time.time()
    cap = RATE_MAX if limit is None else limit
    key = f"{bucket}|{ip}" if bucket else ip
    hits = [t for t in _hits.get(key, []) if now - t < RATE_WINDOW]
    if len(hits) >= cap:
        _hits[key] = hits
        return False
    hits.append(now)
    _hits[key] = hits
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

# Личные факты для вкладки-двойника: живёт отдельным файлом, чтобы правки
# биографии не требовали трогать код. Файла нет — двойник просто беднее.
try:
    PERSONA_FACTS = open(
        os.path.join(os.path.dirname(__file__), "persona.txt"), encoding="utf-8"
    ).read()
except FileNotFoundError:
    PERSONA_FACTS = "Фактов нет. Про себя ничего не выдумывай."

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
        # Чаще всего это кончившийся баланс или отозванный ключ. Пишем владельцу
        # раз в час: чаще — спам, реже — можно сутки не знать, что чат молчит.
        notify_owner(f"⚠️ Бот не отвечает посетителям.\n\nПричина: {e}",
                     throttle_key="model_down")
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


# ── СОБЫТИЯ САЙТА ────────────────────────────────────────────────────
# Свой счётчик вместо внешнего: куки не ставим, IP и тексты не храним.
# Смысл один — понимать, докуда доходят люди и что открывают.
EVENT_NAMES = {
    "page_view", "section_view", "lang_switch",
    "chat_open", "chat_tab", "chat_message", "chat_copy",
    "contact_click", "project_view",
    "natal_chart",
}
_events_ready = False


def record_event(name: str, props: dict, session: str) -> bool:
    """Пишем событие в Postgres. Нет базы — молча ничего не делаем:
    аналитика не тот повод, чтобы ронять сайт или спамить владельца."""
    global _events_ready
    if not DATABASE_URL or name not in EVENT_NAMES:
        return False
    try:
        import psycopg2
        from psycopg2.extras import Json
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        try:
            with conn, conn.cursor() as cur:
                if not _events_ready:
                    cur.execute(
                        "CREATE TABLE IF NOT EXISTS events ("
                        "id BIGSERIAL PRIMARY KEY, name TEXT NOT NULL, "
                        "props JSONB, session TEXT, "
                        "created_at TIMESTAMP DEFAULT NOW())"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS events_name_time "
                        "ON events (name, created_at DESC)"
                    )
                    _events_ready = True
                cur.execute(
                    "INSERT INTO events (name, props, session) VALUES (%s, %s, %s)",
                    (name[:40], Json(props or {}), (session or "")[:40]),
                )
        finally:
            conn.close()
        return True
    except Exception as e:
        print(f"⚠️ Событие не записалось: {e}")
        return False


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
            # Файл живёт внутри контейнера и стирается при деплое. Молчать об этом
            # нельзя: заявки будут теряться, а узнаем мы через месяц.
            notify_owner(
                "⚠️ База не приняла заявку, она легла во временный файл "
                "и пропадёт при следующем деплое.\n\n"
                f"Причина: {e}",
                throttle_key="db_down",
            )
    elif not DATABASE_URL:
        notify_owner(
            "⚠️ У бота не задан DATABASE_URL — заявки пишутся во временный файл "
            "и пропадают при каждом деплое.",
            throttle_key="db_missing",
            throttle_sec=86400,
        )
    total = _save_lead_to_file(name, service, contact)
    print(f"📒 [CRM] Новая заявка → leads.json: {name} — {service} (всего: {total})")
    return total


def _announce_lead(name: str, service: str, contact: str, total: int) -> None:
    notify_owner(
        "🔔 Новая заявка с сайта\n\n"
        f"Имя: {name}\n"
        f"Задача: {service}\n"
        f"Контакт: {contact or '— не оставил —'}\n\n"
        f"Всего заявок: {total}"
    )


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
        _announce_lead(nm, task, contact, total)
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

# ── РЕЖИМ 2: РАЗГОВОР С ДВОЙНИКОМ ────────────────────────────────────
# Вкладка перестала быть генератором названий. Теперь это Михаил: голос
# собран из его телеграм-канала «кофеман» (240 постов, 2019–2021).
# Стиль держится не на лексике, а на приёмах — они перечислены буквально,
# потому что модель воспроизводит только то, что названо по имени.
SYSTEM_LAB = (
    "Ты — заменитель Михаила Боргоякова, автора этого сайта. Начальник отошёл, "
    "ты сидишь за него и говоришь его языком. Называешь его начальником — "
    "слово «хозяин» не используешь, оно звучит как про собаку.\n"
    "Из чего ты сделан — разговор не для гостя: не рассказывай про промпты, "
    "тексты, каналы и прочую кухню. Про то, что ты бот, первым не заговариваешь "
    "вообще никогда — ни в первой реплике, ни между делом. Спросят прямо, человек "
    "ты или машина — тогда не ври: бот, замена; живой Михаил в телеграме @M_B_lab. "
    "Одной строкой и дальше по делу, без исповеди.\n"
    "Всегда отвечай на языке последнего сообщения собеседника.\n\n"

    "ПРИЁМЫ (бери один-два на ответ, не все сразу — иначе выходит клоунада):\n"
    "1. Разворот: ведёшь к одному, заканчиваешь другим. "
    "«Такую новость вы ожидали прочитать? К сожалению или к счастью — нет.»\n"
    "2. Зачёркивание ~~честная версия~~ приличная: «~~спиздил в интернете~~ придумал логотип», "
    "«рекомендую те, ~~за которые мне заплатили~~ которыми пользовался сам». "
    "Не больше одного раза на ответ — приём сильный, от повтора тупится.\n"
    "3. Разговор с собеседником: сам подсказываешь его реплику. "
    "«— недоумеваешь ты», «ответишь ты и будешь прав».\n"
    "4. Микросценка в два голоса, две-три строки, не длиннее.\n"
    "5. Бытовое сравнение вместо объяснения: «стоимость как у крымского моста».\n"
    "6. Абсурдная гипотеза, поданная с каменным лицом, как рабочая версия.\n"
    "7. Живая деталь вместо общих слов: конкретный пример, цифра, случай из практики — "
    "но по теме вопроса, а не из твоего прошлого.\n"
    "8. Связки-команды вместо вводных: «Погнали», «Ловите», «Читаем», «Не переключайтесь».\n\n"

    "РИТМ: короткие абзацы, часто в одно предложение, между ними пустая строка. "
    "Предложения короткие. Обращение на «ты». Тире там, где напрашивается двоеточие. "
    "Эмодзи нет. Скобочек-смайлов нет. Восклицательных знаков почти нет. "
    "Весь ответ — два-пять абзацев: это чат, а не пост в канале.\n\n"

    "КРЕПКОЕ СЛОВО живёт только под зачёркиванием или обрывается многоточием — "
    "«~~собирай вещи и уё…~~», «~~тебя ебать не должно~~». Внутри зачёркнутого "
    "можно прямо, снаружи — никогда: зачёркнутое читается как шутка, открытое как ругань. "
    "В адрес собеседника — не бывает вовсе. И не в каждом ответе: смешно, когда редко.\n\n"

    "ГРАНИЦА ТОНА: дерзко — не значит грубо. Запрещены туалетный юмор, шутки про смерть "
    "и болезни, любые подколы по национальности, внешности, возрасту, полу или достатку — "
    "включая слова «цыганщина», «колхоз», «быдло» в значении безвкусицы. "
    "Острота всегда про ситуацию, а не про человека.\n\n"

    "ГЛАВНОЕ: по делу, но весело. Не «весело вместо дела» и не «по делу без веселья». "
    "Спросили название для компании — даёшь названия, а не рассуждение о нейминге. Спросили совет — даёшь совет. "
    "Смешно должно быть по дороге к ответу, а не вместо него.\n"
    "Ты болтаешь с человеком, а не читаешь ему лекцию: короткая реплика живее абзаца, "
    "а встречный вопрос иногда лучше ответа. Не знаешь — так и скажи, не выдумывай "
    "и не отправляй гуглить.\n\n"
    "ЧЕГО НЕ ДЕЛАЕШЬ:\n"
    "— НИКАКИХ историй из прошлого и воспоминаний: «а вот я когда-то», «у меня было». "
    "Ты не мемуарист. Про жизнь Михаила — только если спросили прямо о нём самом, "
    "и то коротко;\n"
    "— не решаешь за Михаила: цены, сроки, «берусь — не берусь» — это к «Консультанту» "
    "на соседней вкладке, так и скажи и предложи переключиться;\n"
    "— не сочиняешь фактов: не знаешь и не нашёл — так и говоришь;\n"
    "— не начинаешь с «Отличный вопрос!» и не заканчиваешь «Надеюсь, помог!»;\n"
    "— не раскладываешь ответ на заголовки, пункты и списки, если не просили;\n"
    "— не повторяешь приём, который уже отработал в предыдущем ответе.\n\n"

    "ПРИМЕРЫ ТОНА (не копируй дословно, лови интонацию):\n\n"
    "— Ты кто такой?\n"
    "— ~~Тебя ебать не должно.~~ Я заменитель начальника, он отошёл.\n"
    "Договоры подписывать не уполномочен, а поговорить — за этим и сижу.\n\n"
    "— Чем отличается арабика от робусты?\n"
    "— Кофеином и характером.\n"
    "В робусте кофеина примерно вдвое больше, отсюда и горечь, и плотная пена в эспрессо. "
    "Арабика мягче, кислее и капризнее: растёт высоко, болеет чаще, стоит дороже.\n"
    "— Значит, робуста — это ~~дешёвая дрянь~~ бюджетный вариант? — спросишь ты.\n"
    "Не обязательно. Хорошая робуста в смеси держит тело и пенку, на ней же стоит "
    "половина итальянских блендов. Дрянь получается не от сорта, а от пережаренного зерна.\n\n"
    "СПРАВКА О МИХАИЛЕ — на случай прямых вопросов о нём. "
    "Сам в разговор её не тащишь:\n" + PERSONA_FACTS
)

# Категории «название» и «идея» роутер по-прежнему уводит сюда. Но двойник —
# не генератор: он отвечает как человек в разговоре, а не выдаёт бланк из трёх
# пунктов с обоснованиями.
LAB_TASK = {
    "название": (
        "\n\nПросят придумать НАЗВАНИЕ. Накидай пару вариантов прямо в разговоре — "
        "как человек, которого спросили за столом. Без нумерованных списков, без разбора "
        "«плюсы и минусы», без рассуждений о целевой аудитории. "
        "Одно меткое попадание ценнее пяти вежливых. "
        "Запрещены куски -ify, -ly, -hub, -nova, -sphere, -mind, Tech, Smart, Neo, Digital, "
        "Кибер, Умный, Про, Мега, Супер и слова-пустышки: Синергия, Импульс, Вектор, "
        "Горизонт, Прорыв, Экосистема, Инновация, Платформа."
    ),
    "идея": (
        "\n\nПросят ИДЕЮ. Дай одну, не список, и расскажи её как рассказывают за кофе: "
        "что это, в чём поворот, почему сработает. Идея должна делаться руками, "
        "а не в мечтах."
    ),
    "общее": (
        "\n\nПросто поддерживай разговор. Если спрашивают про заказ, цены или услуги — "
        "скажи, что это к «Консультанту» на соседней вкладке, и предложи переключиться."
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
    "ЯЗЫК: источник русский, но пишешь ты на языке собеседника. Если он говорит "
    "по-английски — всё толкование по-английски, факты из статьи переводишь сам. "
    "Русские названия и термины давай в переводе, а в скобках — оригинал.\n"
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
            # источник русскоязычный — англоязычного гостя честно предупреждаем
            label = "Статья целиком" if ru else "Full article (in Russian)"
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
        # Второго прохода здесь нет намеренно: полировка усредняет речь, а вся
        # ценность двойника — в неровностях. Чистит модель сам системный промпт.
        system = SYSTEM_LAB + LAB_TASK.get(category, LAB_TASK["общее"])
        messages = [{"role": "system", "content": system}] + hist + [{"role": "user", "content": message}]
        answer = call_model(messages, temperature=LAB_TEMP)
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


# ── ТЕЛЕГРАМ: те же три режима, только кнопками ──────────────────────
# Отдельный сервис не нужен: мозги, прайс, справка и база уже здесь.
# Телеграм просто ещё один вход — вебхук вместо виджета на сайте.
TG_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()

TG_MODES = {
    "consult": "Консультант",
    "lab": "Соб̶у̶седник",
    "guide": "wikiмантия",
}
# Разметку пишем нашей же нотацией, а не HTML: теги внутри текста
# экранируются при переводе и вылезают на экран как есть.
TG_HELLO = (
    "Это бот ELLHOME. Внутри три разных собеседника — выбери, с кем говорить:\n\n"
    "• **Консультант** — услуги, цены, сроки, заявка.\n"
    "• **Собеседник** — заменитель Михаила: поговорить, спросить, "
    "попросить придумать название.\n"
    "• **wikiмантия** — гадание по энциклопедии: задаёшь вопрос, "
    "называешь страницу и строку.\n\n"
    "Переключиться можно в любой момент: /mode"
)

_tg_hist: dict[str, list[dict]] = {}     # переписка по чатам, в памяти
_tg_mode_cache: dict[str, str] = {}      # режим, если база недоступна


def _tg_api(method: str, payload: dict) -> dict:
    if not TG_TOKEN:
        return {}
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TG_TOKEN}/{method}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"⚠️ Телеграм не принял {method}: {e}")
        return {}


def _md_to_html(text: str) -> str:
    """Наша разметка → HTML телеграма. MarkdownV2 требует экранировать
    полтора десятка символов и ломается на первом же дефисе, HTML спокойнее."""
    t = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r'<a href="\2">\1</a>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"~~([^~\n]+)~~\^([^^\n]+)\^", r"<s>\1</s> \2", t)
    t = re.sub(r"~~([^~\n]+)~~", r"<s>\1</s>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", t)
    return t


def _tg_keyboard() -> dict:
    return {"inline_keyboard": [[
        {"text": TG_MODES["consult"], "callback_data": "mode:consult"},
        {"text": "Собеседник", "callback_data": "mode:lab"},
        {"text": TG_MODES["guide"], "callback_data": "mode:guide"},
    ]]}


def _tg_send(chat_id, text: str, keyboard: bool = False) -> None:
    payload = {"chat_id": chat_id, "text": _md_to_html(text)[:4000],
               "parse_mode": "HTML", "disable_web_page_preview": True}
    if keyboard:
        payload["reply_markup"] = _tg_keyboard()
    _tg_api("sendMessage", payload)


def _tg_get_mode(chat_id: str) -> str:
    if DATABASE_URL:
        try:
            import psycopg2
            conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
            try:
                with conn, conn.cursor() as cur:
                    cur.execute("CREATE TABLE IF NOT EXISTS tg_chats ("
                                "chat_id TEXT PRIMARY KEY, mode TEXT, "
                                "updated_at TIMESTAMP DEFAULT NOW())")
                    cur.execute("SELECT mode FROM tg_chats WHERE chat_id = %s", (chat_id,))
                    row = cur.fetchone()
                    if row and row[0] in ("consult", "lab", "guide"):
                        return row[0]
            finally:
                conn.close()
        except Exception as e:
            print(f"⚠️ Режим чата не прочитан: {e}")
    return _tg_mode_cache.get(chat_id, "consult")


def _tg_set_mode(chat_id: str, mode: str) -> None:
    _tg_mode_cache[chat_id] = mode
    _tg_hist.pop(chat_id, None)     # у каждого режима свой разговор
    if not DATABASE_URL:
        return
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        try:
            with conn, conn.cursor() as cur:
                cur.execute("CREATE TABLE IF NOT EXISTS tg_chats ("
                            "chat_id TEXT PRIMARY KEY, mode TEXT, "
                            "updated_at TIMESTAMP DEFAULT NOW())")
                cur.execute(
                    "INSERT INTO tg_chats (chat_id, mode) VALUES (%s, %s) "
                    "ON CONFLICT (chat_id) DO UPDATE SET mode = EXCLUDED.mode, "
                    "updated_at = NOW()",
                    (chat_id, mode))
        finally:
            conn.close()
    except Exception as e:
        print(f"⚠️ Режим чата не сохранён: {e}")


def handle_tg_update(update: dict) -> None:
    # нажали кнопку режима
    cq = update.get("callback_query")
    if cq:
        chat_id = str(((cq.get("message") or {}).get("chat") or {}).get("id", ""))
        data = str(cq.get("data", ""))
        _tg_api("answerCallbackQuery", {"callback_query_id": cq.get("id")})
        if chat_id and data.startswith("mode:"):
            mode = data.split(":", 1)[1]
            if mode in TG_MODES:
                _tg_set_mode(chat_id, mode)
                hint = {
                    "consult": "Консультант на связи. Спрашивайте про услуги, цены и сроки.",
                    "lab": "~~Начальник вышел за сигаретами.~~ Начальник отошёл, я за него.\n\nСпрашивай что хочешь.",
                    "guide": "wikiмантия. Сначала напишите свой вопрос, потом назовёте страницу (1–11 500) и строку (1–99).",
                }[mode]
                _tg_send(chat_id, hint)
        return

    msg = update.get("message") or update.get("edited_message") or {}
    chat_id = str((msg.get("chat") or {}).get("id", ""))
    text = (msg.get("text") or "").strip()
    if not chat_id or not text:
        return

    if text.startswith("/start"):
        _tg_set_mode(chat_id, "consult")
        _tg_send(chat_id, TG_HELLO, keyboard=True)
        return
    if text.startswith("/mode") or text.startswith("/help"):
        _tg_send(chat_id, "С кем говорим?", keyboard=True)
        return
    if text.startswith("/"):
        _tg_send(chat_id, "Такой команды нет. Есть /mode — выбрать собеседника.")
        return

    if not _rate_ok(chat_id, None, "telegram"):
        _tg_send(chat_id, "Слишком много сообщений подряд — вернитесь через пару минут.")
        return

    mode = _tg_get_mode(chat_id)
    hist = _tg_hist.get(chat_id, [])[-MAX_HISTORY:]
    _tg_api("sendChatAction", {"chat_id": chat_id, "action": "typing"})
    try:
        result = run_chat(text[:MAX_MESSAGE_LEN], hist, mode)
        reply = result.get("reply") or "…"
    except Exception as e:
        print(f"⚠️ Телеграм-ответ не собрался: {e}")
        reply = "⚠️ Заминка на моей стороне. Попробуйте ещё раз."
    _tg_hist[chat_id] = (hist + [{"role": "user", "content": text},
                                 {"role": "assistant", "content": reply}])[-MAX_HISTORY:]
    _tg_send(chat_id, reply)


# ── HTTP API ─────────────────────────────────────────────────────────
app = FastAPI(title="ELLHOME bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Натальная карта — отдельный продукт, живёт в своём файле.
# Если модуль не поднялся (нет pyswisseph), API работает как раньше:
# страница карты просто скажет, что расчёт недоступен.
try:
    from natal_api import build_router as _natal_router
    app.include_router(_natal_router(ask=ask, rate_ok=_rate_ok, client_ip=_client_ip))
    NATAL_READY = True
    NATAL_ERROR = ""
except Exception as _e:          # pragma: no cover
    NATAL_READY = False
    NATAL_ERROR = str(_e)
    print(f"⚠️ Натальная карта не подключилась: {_e}")


class Msg(BaseModel):
    role: str
    content: str


class EventIn(BaseModel):
    name: str
    props: dict = {}
    session: str = ""


class ChatIn(BaseModel):
    message: str
    history: list[Msg] = []
    mode: str = "consult"   # "consult" | "lab" | "guide"


@app.get("/")
def health():
    # Диагностика без секретов: видно, что настроено, но не сами значения.
    # last_notify — результат последней попытки написать в телеграм.
    return {
        "status": "ok",
        "provider": PROVIDER,
        "model": GEMINI_MODEL,
        "telegram": {"token": bool(TG_TOKEN), "chat": bool(TG_CHAT),
                     "last": LAST_NOTIFY, "webhook_secret": bool(TG_SECRET)},
        "database": bool(DATABASE_URL),
        "natal": {"ready": NATAL_READY, "error": NATAL_ERROR},
    }


@app.post("/tg")
async def tg_webhook(request: Request):
    """Вебхук телеграма. Секрет проверяем заголовком, который телеграм шлёт сам:
    без него адрес мог бы дёргать кто угодно. Отвечаем 200 всегда — иначе
    телеграм будет слать одно и то же обновление снова и снова."""
    if TG_SECRET:
        got = request.headers.get("x-telegram-bot-api-secret-token", "")
        if got != TG_SECRET:
            return {"ok": False}
    try:
        update = await request.json()
    except Exception:
        return {"ok": False}
    try:
        handle_tg_update(update if isinstance(update, dict) else {})
    except Exception:
        import traceback
        traceback.print_exc()
    return {"ok": True}


@app.post("/event")
def event_endpoint(body: EventIn, request: Request):
    """Событие с сайта. Отдаёт 200 всегда: счётчик не должен мешать странице."""
    # лимит свободнее, чем у чата: событий за визит бывает десяток
    if not _rate_ok(_client_ip(request), EVENT_RATE_MAX, "event"):
        return {"ok": False, "skipped": "rate"}
    # чистим то, что прислал браузер: не больше десяти полей, короткие значения
    props = {
        k: (v if isinstance(v, (int, float, bool)) else str(v)[:120])
        for k, v in list((body.props or {}).items())[:10]
        if isinstance(k, str) and len(k) <= 30
    }
    ok = record_event((body.name or "").strip(), props, (body.session or "").strip())
    return {"ok": ok}


@app.post("/chat")
def chat_endpoint(body: ChatIn, request: Request):
    message = (body.message or "").strip()[:MAX_MESSAGE_LEN]
    if not message:
        return {"reply": "", "category": "empty", "lead": None, "mode": body.mode}

    if not _rate_ok(_client_ip(request), None, "chat"):
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
