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
# Запись из браузера уходит модели как есть: Gemini понимает и webm/opus
# (Chrome, Android), и m4a/aac (Safari, iPhone). Распознавание в самом
# браузере не годится: в Firefox и Яндекс.Браузере его нет, а в Chrome оно
# «исправляет» ошибки — человек говорит «she go», а видит «she goes»,
# и собеседнику нечего поправить.
HEAR_MODELS = [m for m in (os.getenv("GEMINI_MODEL"), "gemini-2.5-flash") if m]
HEAR_MAX = 2_000_000             # ~минута речи в opus; больше — это уже не реплика
HEAR_TYPES = {"audio/webm", "audio/ogg", "audio/mp4", "audio/m4a", "audio/x-m4a", "audio/aac",
              "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave", "audio/flac"}
HEAR_PROMPT = (
    "Transcribe this recording word for word. The speaker is a Russian adult learning "
    "English. Keep every grammar mistake exactly as spoken — do NOT correct anything, "
    "the mistakes are what a tutor needs to see. If a phrase is in Russian, write it in "
    "Russian (Cyrillic). If a word is unclear, write your best guess. If there is no "
    "speech at all, return an empty line. Return only the transcript, no quotes, no comments."
)


def _hear_request(model: str, audio: bytes, mime: str, key: str, opener=None) -> str:
    body = {
        "contents": [{"parts": [
            {"text": HEAR_PROMPT},
            {"inlineData": {"mimeType": mime, "data": base64.b64encode(audio).decode()}},
        ]}],
        "generationConfig": {"temperature": 0},
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with (opener or urllib.request.urlopen)(req, timeout=40) as r:
        d = json.loads(r.read())
    parts = d["candidates"][0]["content"].get("parts") or []
    return "".join(p.get("text", "") for p in parts)


def hear(audio: bytes, mime: str, opener=None) -> tuple[str, str]:
    """→ (текст, ошибка). Ошибка — короткая фраза для человека."""
    mime = (mime or "").split(";")[0].strip().lower()
    if mime not in HEAR_TYPES:
        return "", "Этот формат записи не поддерживается."
    if len(audio) < 800:
        return "", "Запись слишком короткая — нажмите и скажите фразу."
    if len(audio) > HEAR_MAX:
        return "", "Слишком длинная запись. Скажите короче, одной-двумя фразами."
    key = os.getenv("GOOGLE_API_KEY") or ""
    if not key or "..." in key:
        return "", "Распознавание сейчас недоступно."
    for model in HEAR_MODELS:
        try:
            t = _hear_request(model, audio, mime, key, opener)
        except Exception as e:
            print(f"⚠️ Распознавание {model} не сработало: {e}")
            continue
        t = re.sub(r"\s+", " ", t).strip().strip('"«»').strip()[:MAX_TEXT]
        if not t:
            return "", "Не расслышал. Попробуйте ещё раз, чуть ближе к микрофону."
        return t, ""
    return "", "Распознавание сейчас недоступно. Попробуйте ещё раз."


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
        text, err = hear(audio, request.headers.get("content-type", ""))
        return {"error": err} if err else {"text": text}

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
