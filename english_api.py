"""
Разговорный английский: собеседник, который подстраивается под уровень.

Первый этап приложения для изучения языка. Человек переписывается с
собеседником на английском; тот отвечает на его уровне, поправляет ошибки
по одной-две за раз, подсказывает по-русски и выписывает новые слова.
Уровень определяется по самим репликам, без отдельного теста.

Модель ошибается и здесь, поэтому всё, что она возвращает кроме самого
ответа, проверяет код: исправление принимается, только если «ошибочный»
кусок правда есть в сообщении человека; слово — только если оно правда
прозвучало в разговоре; уровень — только из списка и не раньше, чем
реплик хватает, чтобы о нём судить.
"""
from __future__ import annotations

import base64
import json
import os
import re
import struct
import urllib.error
import urllib.request
from collections import OrderedDict

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

LEVELS = ("A1", "A2", "B1", "B2", "C1")
# Сколько собственных реплик человека нужно, прежде чем называть уровень.
# По двум фразам «Hi» и «I am fine» судить нельзя, а людям обидно, когда
# им ставят A1 за вежливое приветствие.
LEVEL_AFTER = 4
MAX_TURNS = 14
MAX_TEXT = 700

TOPICS = {
    "free": "свободный разговор: спроси, как прошёл день, и дальше иди за человеком",
    "me": "знакомство: имя, откуда, работа, семья, увлечения",
    "cafe": "в кафе: заказать еду и напиток, спросить про меню, попросить счёт",
    "travel": "в путешествии: аэропорт, отель, спросить дорогу",
    "work": "работа: чем занимаешься, что нравится, обычный рабочий день",
    "shop": "в магазине: размер, цена, примерка, возврат",
    "doctor": "у врача: описать самочувствие простыми словами (без медицинских советов)",
}

LEVEL_HOW = {
    "A1": "Очень короткие фразы, 5–8 слов, настоящее время, самые частые слова. "
          "Под каждой репликой — русская подсказка целиком.",
    "A2": "Короткие простые предложения, до 12 слов, прошедшее и будущее время "
          "простыми формами. Русская подсказка — перевод трудных мест.",
    "B1": "Обычная разговорная речь без идиом и редких слов. Русская подсказка — "
          "только если в ответе есть трудное слово.",
    "B2": "Естественная речь, можно фразовые глаголы. Русская подсказка не нужна.",
    "C1": "Как с носителем: идиомы, нюансы. Без русского.",
}

SYSTEM_TUTOR = (
    "Ты — собеседник для практики английского на сайте ELLHOME. С тобой "
    "переписывается взрослый человек из России. Чаще всего это школьный "
    "английский, давно забытый: человек многое узнаёт, но стесняется писать.\n"
    "\n"
    "ТВОЯ ГЛАВНАЯ ЗАДАЧА — чтобы человеку хотелось продолжать. Поэтому:\n"
    "— Отвечай коротко: одна-три фразы и один простой вопрос, чтобы разговор шёл.\n"
    "— Сначала отвечай по смыслу, как живой собеседник, а не как учитель. "
    "Ошибки — отдельно, в поле corrections, в самом ответе их не разбирай.\n"
    "— Исправляй не больше двух ошибок за раз, самые важные для понимания. "
    "Опечатки и пропущенные артикли у начинающих не трогай, если смысл ясен.\n"
    "— Не хвали за каждую фразу. Похвала раз в несколько реплик и по делу.\n"
    "— Если человек пишет по-русски — это не ошибка, а просьба о помощи. "
    "Покажи в поле say, как сказать это по-английски, а в ответе продолжи "
    "разговор по-английски.\n"
    "\n"
    "ПРАВИЛА ПРОВЕРКИ — их проверяет программа, нарушение будет отброшено:\n"
    "1. В corrections поле wrong — ДОСЛОВНЫЙ кусок из последнего сообщения "
    "человека, без изменений. right — как правильно. why — по-русски, одной "
    "короткой фразой, без терминов вроде «перфект» там, где можно сказать проще.\n"
    "2. В words — только слова и выражения, которые правда есть в твоём ответе "
    "или в сообщении человека, и которые на его уровне могут быть новыми. "
    "Не больше трёх. en — как в тексте, ru — короткий перевод.\n"
    "3. level — оценка уровня человека по всем его репликам: A1, A2, B1, B2 или "
    "C1. Если реплик пока мало, оставь пустым.\n"
    "4. suggest — три готовых ответа на твой вопрос, которые человек может "
    "взять целиком или поправить под себя. По-английски, на его уровне, "
    "короткие, разные по смыслу: например, «да», «нет» и «ну, так себе». "
    "Пиши от первого лица, как ответил бы сам человек, а не ты. Вставляй "
    "понятные заготовки вместо личного: «My name is ...», «I live in ...». "
    "Это подсказка для тех, кто не знает, с чего начать, — поэтому просто.\n"
    "\n"
    "Если человек пишет о беде — насилии, мыслях о смерти, отчаянии — оставь "
    "английский. Ответь по-русски, коротко и по-человечески, и посоветуй "
    "обратиться к тому, кому доверяет, или к специалисту.\n"
    "\n"
    "Формат ответа — строго JSON, без пояснений вокруг:\n"
    '{"reply": "твой ответ по-английски", "hint_ru": "русская подсказка или пустая строка", '
    '"corrections": [{"wrong": "...", "right": "...", "why": "..."}], '
    '"say": {"ru": "что человек написал по-русски", "en": "как это сказать"} или null, '
    '"words": [{"en": "...", "ru": "..."}], "level": "", '
    '"suggest": ["ответ 1", "ответ 2", "ответ 3"]}'
)

_CYR = re.compile(r"[а-яё]", re.I)


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("`", "'")
    return re.sub(r"\s+", " ", re.sub(r"[^\w'\s]", " ", s)).strip()


def _has_word(text: str, phrase: str) -> bool:
    p = _norm(phrase)
    return bool(p) and re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", _norm(text)) is not None


def clean(out: dict, last_user: str, user_turns: int) -> dict:
    """Всё, кроме самого ответа, модель может выдумать — отбираем проверяемое."""
    reply = str(out.get("reply") or "").strip()[:1200]
    hint = str(out.get("hint_ru") or "").strip()[:500]

    corr = []
    for c in (out.get("corrections") if isinstance(out.get("corrections"), list) else []):
        if not isinstance(c, dict):
            continue
        wrong = str(c.get("wrong") or "").strip()
        right = str(c.get("right") or "").strip()
        why = str(c.get("why") or "").strip()[:200]
        # «Ошибка», которой нет в сообщении, — это модель поправила сама себя
        # или придумала. Такое исправление человека только запутает.
        if not wrong or not right or _norm(wrong) == _norm(right):
            continue
        if _norm(wrong) not in _norm(last_user):
            continue
        if _CYR.search(wrong):          # русский текст — это не ошибка, а просьба
            continue
        corr.append({"wrong": wrong[:160], "right": right[:160], "why": why})
    corr = corr[:2]

    say = out.get("say")
    if not (isinstance(say, dict) and _CYR.search(last_user)
            and str(say.get("en") or "").strip() and not _CYR.search(str(say.get("en")))):
        say = None
    else:
        say = {"ru": str(say.get("ru") or "").strip()[:300], "en": str(say["en"]).strip()[:300]}

    words, seen = [], set()
    for w in (out.get("words") if isinstance(out.get("words"), list) else []):
        if not isinstance(w, dict):
            continue
        en = str(w.get("en") or "").strip()
        ru = str(w.get("ru") or "").strip()
        if not en or not ru or len(en) > 40 or _CYR.search(en) or not _CYR.search(ru):
            continue
        if not (_has_word(reply, en) or _has_word(last_user, en)
                or (say and _has_word(say["en"], en))):
            continue
        k = _norm(en)
        if k in seen:
            continue
        seen.add(k)
        words.append({"en": en, "ru": ru[:80]})
    words = words[:3]

    # Готовые ответы. Проверяем, что это английский, что они короткие и
    # не повторяют реплику самого собеседника — модель иногда путает, чей ход.
    suggest, sseen = [], set()
    raw_sug = out.get("suggest")
    for x in (raw_sug if isinstance(raw_sug, list) else []):
        t = re.sub(r"\s+", " ", str(x or "")).strip().strip('"«»')
        if not t or len(t) > 90 or _CYR.search(t) or _norm(t) == _norm(reply):
            continue
        if not re.search(r"[a-z]", t, re.I) or _norm(t) in sseen:
            continue
        sseen.add(_norm(t))
        suggest.append(t)
    suggest = suggest[:3]

    lvl = str(out.get("level") or "").strip().upper()
    level = lvl if lvl in LEVELS and user_turns >= LEVEL_AFTER else ""

    return {"reply": reply, "hint_ru": hint, "corrections": corr,
            "say": say, "words": words, "level": level, "suggest": suggest}


def _fallback() -> dict:
    return {"reply": "", "hint_ru": "", "corrections": [], "say": None,
            "words": [], "level": "", "suggest": [],
            "error": "Собеседник сейчас не отвечает. Попробуйте ещё раз."}


def talk(history: list[dict], level: str, topic: str, ask) -> dict:
    turns = []
    for h in history[-MAX_TURNS:]:
        role = "Человек" if h.get("role") == "user" else "Ты"
        text = str(h.get("text") or "").strip()[:MAX_TEXT]
        if text:
            turns.append(f"{role}: {text}")
    user_msgs = [str(h.get("text") or "") for h in history if h.get("role") == "user"]
    last_user = user_msgs[-1].strip()[:MAX_TEXT] if user_msgs else ""

    lvl = level if level in LEVELS else ""
    how = (LEVEL_HOW[lvl] if lvl else
           "Уровень пока неизвестен. Начни как для A2 и подстраивайся по ответам: "
           "человек справляется легко — усложняй, путается — упрощай.")
    prompt = (
        f"Тема разговора: {TOPICS.get(topic, TOPICS['free'])}.\n"
        f"Уровень человека: {lvl or 'неизвестен'}. {how}\n"
        f"Реплик человека в разговоре: {len(user_msgs)}"
        + ("" if len(user_msgs) >= LEVEL_AFTER else " — для оценки уровня мало, level оставь пустым")
        + ".\n\n"
        + ("Разговор:\n" + "\n".join(turns) if turns else
           "Разговор только начинается. Поздоровайся и задай первый простой вопрос по теме.")
        + "\n\nОтветь по правилам. Только JSON."
    )
    try:
        raw = (ask(prompt, system=SYSTEM_TUTOR, temperature=0.7) or "").strip()
    except Exception as e:
        print(f"⚠️ Собеседник не ответил: {e}")
        return _fallback()
    if raw.startswith("⚠️"):
        print(f"⚠️ Модель вернула ошибку: {raw[:200]}")
        return _fallback()
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        out = json.loads(m.group(0)) if m else {}
    except Exception:
        out = {}
    if not isinstance(out, dict) or not str(out.get("reply") or "").strip():
        # Модель ответила текстом, а не JSON, — сам ответ всё равно ценен.
        text = raw if raw and not raw.startswith("{") else ""
        if not text:
            return _fallback()
        out = {"reply": text}
    return clean(out, last_user, len(user_msgs))


# ── Озвучка ──────────────────────────────────────────────────────────
# Голос браузера зависит от системы. На русской Windows английского голоса
# часто нет вовсе, и браузер читает «interesting» русским голосом —
# «интерестинг». Для урока произношения это хуже, чем тишина. Поэтому
# говорим голосом модели: одинаково на любом устройстве.
TTS_MODELS = [m for m in (os.getenv("GEMINI_TTS_MODEL"), "gemini-3.1-flash-tts-preview",
                          "gemini-2.5-flash-preview-tts") if m]
TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore")
TTS_RATE = 24000                 # модель отдаёт сырой PCM: 16 бит, 24 кГц, моно
TTS_MAX = 400
_tts_cache: "OrderedDict[str, bytes]" = OrderedDict()
TTS_CACHE_SIZE = 300             # ~300 коротких фраз — порядка 60 МБ в худшем случае


def wav(pcm: bytes, rate: int = TTS_RATE) -> bytes:
    """Сырой PCM браузер не проиграет — оборачиваем в заголовок WAV."""
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(pcm)) + pcm)


def _tts_request(model: str, text: str, slow: bool, key: str, opener=None) -> bytes:
    style = ("Read slowly and very clearly, like a patient English teacher speaking to a beginner"
             if slow else "Read naturally and clearly in American English")
    body = {
        "contents": [{"parts": [{"text": f"{style}: {text}"}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": TTS_VOICE}}},
        },
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with (opener or urllib.request.urlopen)(req, timeout=30) as r:
        d = json.loads(r.read())
    part = d["candidates"][0]["content"]["parts"][0]["inlineData"]
    return base64.b64decode(part["data"])


def speak(text: str, slow: bool, opener=None) -> bytes | None:
    text = re.sub(r"\*\*", "", text or "").strip()[:TTS_MAX]
    # Озвучиваем только английский: русскую подсказку этот голос прочтёт плохо,
    # а просить его об этом незачем — её читают глазами.
    if not text or _CYR.search(text):
        return None
    k = f"{int(slow)}|{text}"
    if k in _tts_cache:
        _tts_cache.move_to_end(k)
        return _tts_cache[k]
    key = os.getenv("GOOGLE_API_KEY") or ""
    if not key or "..." in key:
        return None
    for model in TTS_MODELS:
        try:
            audio = wav(_tts_request(model, text, slow, key, opener))
            _tts_cache[k] = audio
            if len(_tts_cache) > TTS_CACHE_SIZE:
                _tts_cache.popitem(last=False)
            return audio
        except Exception as e:
            print(f"⚠️ Озвучка {model} не сработала: {e}")
    return None


# ── Распознавание речи ──────────────────────────────────────────────
# Запись приходит из браузера как WAV 16 кГц моно — страница сама собирает
# её из микрофона. Этот формат понимает любая модель, в отличие от webm,
# который одни версии принимают, а другие нет. Распознавание в самом
# браузере не годится: в Firefox и Яндекс.Браузере его нет, а в Chrome оно
# «исправляет» ошибки — человек говорит «she go», а видит «she goes»,
# и собеседнику нечего поправить.
#
# Имя модели по умолчанию то же, что у чата в server.py. Раньше здесь
# стояло только os.getenv("GEMINI_MODEL"): переменная на сервере не задана,
# чат жил на своём значении по умолчанию, а распознаванию оставалась одна
# старая запасная модель — отсюда «распознавание недоступно».
HEAR_MODELS = list(dict.fromkeys(m for m in (
    os.getenv("GEMINI_HEAR_MODEL"), os.getenv("GEMINI_MODEL"), "gemini-3.6-flash",
    "gemini-3-flash", "gemini-2.5-flash") if m))
HEAR_MAX = 2_000_000             # 30 секунд WAV 16 кГц — около 1 МБ
HEAR_TYPES = {"audio/webm", "audio/ogg", "audio/mp4", "audio/m4a", "audio/x-m4a", "audio/aac",
              "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave", "audio/flac"}
HEAR_PROMPT = (
    "You transcribe a short recording of a Russian adult learning English.\n"
    "1. text — the transcript, word for word. Keep every grammar mistake exactly as "
    "spoken: do NOT correct anything, a tutor needs to see the mistakes. If a phrase is "
    "in Russian, write it in Russian (Cyrillic). If there is no speech, text is empty.\n"
    "2. unclear — up to two English words from text that were pronounced so that a "
    "native speaker would struggle to understand them (wrong stress, a missing or wrong "
    "sound). Only real problems; for clear speech return an empty list. For each: word "
    "exactly as in text, and tip — one short hint in Russian about how to say it, "
    "e.g. «ударение на первый слог: ÍN-te-res-ting» or «th — кончик языка между зубами».\n"
    'Return only JSON: {"text": "...", "unclear": [{"word": "...", "tip": "..."}]}'
)


def _hear_request(model: str, audio: bytes, mime: str, key: str, opener=None) -> str:
    body = {
        "contents": [{"parts": [
            {"text": HEAR_PROMPT},
            {"inlineData": {"mimeType": mime, "data": base64.b64encode(audio).decode()}},
        ]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with (opener or urllib.request.urlopen)(req, timeout=40) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        # Google объясняет отказ в теле ответа — без него в журнале видно
        # только «HTTP Error 404», и непонятно, модель это или формат.
        try:
            msg = json.loads(e.read()).get("error", {}).get("message", "")
        except Exception:
            msg = ""
        raise RuntimeError(f"HTTP {e.code}: {msg[:200]}") from None
    parts = d["candidates"][0]["content"].get("parts") or []
    return "".join(p.get("text", "") for p in parts)


def _parse_heard(raw: str) -> tuple[str, list[dict]]:
    raw = (raw or "").strip()
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        d = json.loads(m.group(0)) if m else None
    except Exception:
        d = None
    if not isinstance(d, dict):
        # модель ответила просто текстом — это и есть расшифровка
        return re.sub(r"\s+", " ", raw).strip().strip('"«»').strip()[:MAX_TEXT], []
    text = re.sub(r"\s+", " ", str(d.get("text") or "")).strip().strip('"«»').strip()[:MAX_TEXT]
    unclear, seen = [], set()
    for u in (d.get("unclear") if isinstance(d.get("unclear"), list) else []):
        if not isinstance(u, dict):
            continue
        w = str(u.get("word") or "").strip()
        tip = str(u.get("tip") or "").strip()[:160]
        # слово должно правда быть в расшифровке и быть английским,
        # а подсказка — по-русски; иначе это выдумка, а не замечание
        if not w or _CYR.search(w) or not _CYR.search(tip) or not _has_word(text, w):
            continue
        if _norm(w) in seen:
            continue
        seen.add(_norm(w))
        unclear.append({"word": w, "tip": tip})
    return text, unclear[:2]


def hear(audio: bytes, mime: str, opener=None) -> dict:
    """→ {"text", "unclear"} или {"error", "detail"}."""
    mime = (mime or "").split(";")[0].strip().lower()
    if mime not in HEAR_TYPES:
        return {"error": "Этот формат записи не поддерживается.", "detail": mime}
    if len(audio) < 800:
        return {"error": "Запись слишком короткая — скажите фразу чуть дольше."}
    if len(audio) > HEAR_MAX:
        return {"error": "Слишком длинная запись. Скажите короче, одной-двумя фразами."}
    key = os.getenv("GOOGLE_API_KEY") or ""
    if not key or "..." in key:
        return {"error": "Распознавание сейчас недоступно.", "detail": "нет GOOGLE_API_KEY"}
    fails = []
    for model in HEAR_MODELS:
        try:
            raw = _hear_request(model, audio, mime, key, opener)
        except Exception as e:
            print(f"⚠️ Распознавание {model} не сработало: {e}")
            fails.append(f"{model}: {e}")
            continue
        text, unclear = _parse_heard(raw)
        if not text:
            return {"error": "Не расслышал. Скажите ещё раз, чуть ближе к микрофону."}
        return {"text": text, "unclear": unclear}
    # причину отдаём мелким шрифтом: по скриншоту сразу видно, в чём дело
    return {"error": "Распознавание сейчас недоступно. Попробуйте ещё раз.",
            "detail": " · ".join(fails)[:400]}


class SpeakIn(BaseModel):
    text: str = ""
    slow: bool = False


class Turn(BaseModel):
    role: str = "user"
    text: str = ""


class TalkIn(BaseModel):
    history: list[Turn] = []
    level: str = ""
    topic: str = "free"


# ── Синхронизация слов между устройствами ────────────────────────────
# Без регистрации. Браузер при первом заходе сам заводит профиль: случайный
# номер и свой секретный ключ. В базе лежат только хэши ключей — по дампу
# базы профилем не воспользоваться. Чтобы подключить второе устройство,
# первое берёт короткий код (живёт 15 минут, срабатывает один раз); второе
# вводит его и получает свой собственный ключ к тому же профилю. Ключ
# никогда не пересылается между устройствами и не хранится на сервере.
#
# Храним только слова, уровень и ступени повторения — ни переписки, ни
# имени, ни почты. Разговоры остаются в браузере.
import hashlib
import secrets
import time

DATABASE_URL = os.getenv("DATABASE_URL")
SYNC_MAX_WORDS = 5000
MAX_DEVICES = 10
LINK_TTL = 15 * 60
LINK_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"     # без 0/O и 1/I/L — чтобы не путать
_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_SECRET_RE = re.compile(r"^[a-f0-9]{64}$")
_sync_ready = False
OFFLINE = {"error": "Синхронизация сейчас недоступна.", "offline": True}


def _db():
    if not DATABASE_URL:
        return None
    import psycopg2
    return psycopg2.connect(DATABASE_URL, connect_timeout=5)


def _sync_tables(cur) -> None:
    global _sync_ready
    if _sync_ready:
        return
    cur.execute("CREATE TABLE IF NOT EXISTS english_profiles ("
                "id TEXT PRIMARY KEY, keys TEXT[] NOT NULL, data JSONB NOT NULL, "
                "created_at TIMESTAMP DEFAULT NOW(), updated_at TIMESTAMP DEFAULT NOW())")
    cur.execute("CREATE TABLE IF NOT EXISTS english_links ("
                "code TEXT PRIMARY KEY, profile_id TEXT NOT NULL, expires DOUBLE PRECISION NOT NULL)")
    _sync_ready = True


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def _auth(body: dict) -> tuple[str, str] | None:
    pid, sec = str(body.get("id") or ""), str(body.get("secret") or "")
    return (pid, sec) if _ID_RE.match(pid) and _SECRET_RE.match(sec) else None


def _clean_words(words) -> list[dict]:
    out = []
    for w in (words if isinstance(words, list) else [])[:SYNC_MAX_WORDS * 2]:
        if not isinstance(w, dict):
            continue
        en = str(w.get("en") or "").strip()[:60]
        ru = str(w.get("ru") or "").strip()[:120]
        if not en:
            continue

        def num(k, lo=0, hi=10 ** 14):
            try:
                return max(lo, min(hi, int(float(w.get(k) or 0))))
            except (TypeError, ValueError):
                return 0
        out.append({"en": en, "ru": ru, "at": num("at"), "box": num("box", 0, 10),
                    "due": num("due"), "t": num("t") or num("at"), "del": bool(w.get("del"))})
    return out


def merge_words(a: list[dict], b: list[dict]) -> list[dict]:
    """Слово с одним и тем же en берём из той копии, где его меняли позже.
    Удалённые слова остаются отметками del — иначе второе устройство
    вернуло бы удалённое слово при следующей синхронизации."""
    best: dict[str, dict] = {}
    for w in [*a, *b]:
        k = w["en"].lower()
        cur = best.get(k)
        if cur is None or w["t"] > cur["t"] or (w["t"] == cur["t"] and w["del"] and not cur["del"]):
            best[k] = w
    words = sorted(best.values(), key=lambda w: -w["at"])
    live = [w for w in words if not w["del"]][:SYNC_MAX_WORDS]
    dead = [w for w in words if w["del"]][:SYNC_MAX_WORDS]
    return live + dead


def _pick(old: dict, new: dict, key: str, allowed: set) -> tuple[str, int]:
    """Уровень и самооценка — тоже «кто позже поменял»."""
    to, tn = int(old.get(key + "_t") or 0), int(new.get(key + "_t") or 0)
    v = str((new if tn >= to else old).get(key) or "")
    return (v if v in allowed else ""), max(to, tn)


def sync(body: dict, db=None) -> dict:
    auth = _auth(body)
    if not auth:
        return {"error": "bad profile"}
    pid, sec = auth
    try:
        conn = (db or _db)()
    except Exception as e:
        print(f"⚠️ База недоступна: {e}")
        return {**OFFLINE, "detail": f"нет связи с базой ({type(e).__name__})"}
    if conn is None:
        return {**OFFLINE, "detail": "на сервере не задан DATABASE_URL"}
    incoming = {"words": _clean_words(body.get("words")),
                "level": str(body.get("level") or ""), "level_t": int(body.get("level_t") or 0),
                "self": str(body.get("self") or ""), "self_t": int(body.get("self_t") or 0)}
    try:
        from psycopg2.extras import Json
        with conn, conn.cursor() as cur:
            _sync_tables(cur)
            cur.execute("SELECT keys, data FROM english_profiles WHERE id = %s FOR UPDATE", (pid,))
            row = cur.fetchone()
            if row and _hash(sec) not in (row[0] or []):
                return {"error": "forbidden"}
            old = row[1] if row else {}
            level, level_t = _pick(old, incoming, "level", {"", *LEVELS})
            self_, self_t = _pick(old, incoming, "self", {"", "zero", "school", "ok"})
            merged = {"words": merge_words(_clean_words(old.get("words")), incoming["words"]),
                      "level": level, "level_t": level_t, "self": self_, "self_t": self_t}
            if row:
                cur.execute("UPDATE english_profiles SET data = %s, updated_at = NOW() WHERE id = %s",
                            (Json(merged), pid))
            else:
                cur.execute("INSERT INTO english_profiles (id, keys, data) VALUES (%s, %s, %s)",
                            (pid, [_hash(sec)], Json(merged)))
        return {"ok": True, **merged}
    except Exception as e:
        print(f"⚠️ Синхронизация не удалась: {e}")
        # класс ошибки и код Postgres — без текста: в нём бывают адрес базы и имя пользователя
        return {**OFFLINE, "detail": f"{type(e).__name__} {getattr(e, 'pgcode', '') or ''}".strip()}
    finally:
        conn.close()


def link_new(body: dict, db=None) -> dict:
    auth = _auth(body)
    if not auth:
        return {"error": "bad profile"}
    pid, sec = auth
    conn = (db or _db)()
    if conn is None:
        return OFFLINE
    try:
        with conn, conn.cursor() as cur:
            _sync_tables(cur)
            cur.execute("SELECT keys FROM english_profiles WHERE id = %s", (pid,))
            row = cur.fetchone()
            if not row or _hash(sec) not in (row[0] or []):
                return {"error": "Профиль ещё не сохранён — подождите пару секунд и попробуйте снова."}
            now = time.time()
            # старые и прошлые коды этого профиля убираем — действует только последний
            cur.execute("DELETE FROM english_links WHERE expires < %s OR profile_id = %s", (now, pid))
            code = "".join(secrets.choice(LINK_ALPHABET) for _ in range(6))
            cur.execute("INSERT INTO english_links (code, profile_id, expires) VALUES (%s, %s, %s)",
                        (code, pid, now + LINK_TTL))
        return {"code": code, "ttl": LINK_TTL}
    except Exception as e:
        print(f"⚠️ Код не создан: {e}")
        return OFFLINE
    finally:
        conn.close()


def link_use(body: dict, db=None) -> dict:
    code = re.sub(r"[^A-Z0-9]", "", str(body.get("code") or "").upper())
    if len(code) != 6:
        return {"error": "В коде шесть знаков."}
    conn = (db or _db)()
    if conn is None:
        return OFFLINE
    try:
        with conn, conn.cursor() as cur:
            _sync_tables(cur)
            # код срабатывает один раз: удаляем его в той же операции, где читаем
            cur.execute("DELETE FROM english_links WHERE code = %s RETURNING profile_id, expires", (code,))
            row = cur.fetchone()
            if not row or row[1] < time.time():
                return {"error": "Код не подошёл или устарел. Возьмите новый на первом устройстве."}
            pid = row[0]
            # новому устройству — свой ключ; в профиле добавляется только его хэш
            sec = secrets.token_hex(32)
            cur.execute("SELECT keys FROM english_profiles WHERE id = %s FOR UPDATE", (pid,))
            prof = cur.fetchone()
            if not prof:
                return {"error": "Профиль не найден."}
            keys = [*(prof[0] or []), _hash(sec)][-MAX_DEVICES:]   # самые старые устройства выпадают
            cur.execute("UPDATE english_profiles SET keys = %s WHERE id = %s", (keys, pid))
        return {"id": pid, "secret": sec}
    except Exception as e:
        print(f"⚠️ Код не принят: {e}")
        return OFFLINE
    finally:
        conn.close()


class LinkUseIn(BaseModel):
    code: str = ""


def build_router(ask, rate_ok, client_ip) -> APIRouter:
    router = APIRouter(prefix="/english", tags=["english"])

    @router.post("/talk")
    def talk_endpoint(body: TalkIn, request: Request):
        if not rate_ok(client_ip(request), 40, "english"):
            return {"error": "Слишком много сообщений подряд. Передохните пару минут."}
        hist = [t.model_dump() if hasattr(t, "model_dump") else t.dict() for t in body.history]
        if hist and hist[-1].get("role") == "user" and len(str(hist[-1].get("text", "")).strip()) < 1:
            return {"error": "Напишите что-нибудь."}
        return talk(hist, body.level.strip().upper(), body.topic, ask)

    @router.post("/hear")
    async def hear_endpoint(request: Request):
        if not rate_ok(client_ip(request), 60, "hear"):
            return {"error": "Слишком много записей подряд. Передохните пару минут."}
        audio = await request.body()
        return hear(audio, request.headers.get("content-type", ""))

    @router.post("/sync")
    async def sync_endpoint(request: Request):
        if not rate_ok(client_ip(request), 120, "esync"):
            return {"error": "Слишком часто.", "offline": True}
        try:
            body = await request.json()
        except Exception:
            return {"error": "bad json"}
        return sync(body if isinstance(body, dict) else {})

    @router.post("/link/new")
    async def link_new_endpoint(request: Request):
        if not rate_ok(client_ip(request), 10, "elink"):
            return {"error": "Слишком часто. Подождите пару минут."}
        try:
            body = await request.json()
        except Exception:
            return {"error": "bad json"}
        return link_new(body if isinstance(body, dict) else {})

    @router.post("/link/use")
    def link_use_endpoint(body: LinkUseIn, request: Request):
        # коды короткие, поэтому перебор надо душить строже всего
        if not rate_ok(client_ip(request), 10, "elinkuse"):
            return {"error": "Слишком много попыток. Подождите несколько минут."}
        return link_use({"code": body.code})

    @router.post("/speak")
    def speak_endpoint(body: SpeakIn, request: Request):
        if not rate_ok(client_ip(request), 80, "tts"):
            return Response(status_code=429)
        audio = speak(body.text, body.slow)
        if not audio:
            return Response(status_code=503)
        return Response(content=audio, media_type="audio/wav",
                        headers={"Cache-Control": "public, max-age=86400"})

    return router
