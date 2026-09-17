"""Натальная карта как продукт: поиск города, расчёт, толкование.

Отдельный файл, чтобы server.py не пух. Подключается одной строкой:

    from natal_api import build_router
    app.include_router(build_router(ask=ask, rate_ok=_rate_ok, client_ip=_client_ip))

Астрономия честная — Swiss Ephemeris. Всё, что модель говорит про характер,
строится только на заранее посчитанных фактах: числа она не выдумывает,
потому что их ей выдают готовыми.
"""
from __future__ import annotations

import os
import re
import json
import datetime as dt
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Request
from pydantic import BaseModel

DATABASE_URL = os.getenv("DATABASE_URL")

GEO_API = "https://geocoding-api.open-meteo.com/v1/search"
GEO_TIMEOUT = 8
GEO_UA = "ELLHOME-natal/1.0 (https://ellhome.github.io/Mikhail.Borgoyakov)"

# Год, раньше которого разговор про «местное время рождения» теряет смысл:
# до этого в ходу было солнечное время, а часовых поясов ещё не завели.
MIN_YEAR = 1900

try:
    import natal as engine
except Exception as e:                      # pragma: no cover
    engine = None
    ENGINE_ERROR = str(e)
else:
    ENGINE_ERROR = ""


# ── База: маленькие помощники, которые молчат, если базы нет ─────────
_db_ready = False


def _conn():
    if not DATABASE_URL:
        return None
    import psycopg2
    return psycopg2.connect(DATABASE_URL, connect_timeout=5)


def _ensure_tables(cur) -> None:
    global _db_ready
    if _db_ready:
        return
    cur.execute("CREATE TABLE IF NOT EXISTS geo_cache ("
                "q TEXT PRIMARY KEY, payload JSONB NOT NULL, "
                "created_at TIMESTAMP DEFAULT NOW())")
    cur.execute("CREATE TABLE IF NOT EXISTS natal_readings ("
                "key TEXT PRIMARY KEY, payload JSONB NOT NULL, "
                "created_at TIMESTAMP DEFAULT NOW())")
    _db_ready = True


def _cache_get(table: str, col: str, key: str):
    try:
        conn = _conn()
        if not conn:
            return None
        try:
            with conn, conn.cursor() as cur:
                _ensure_tables(cur)
                cur.execute(f"SELECT payload FROM {table} WHERE {col} = %s", (key,))
                row = cur.fetchone()
                return row[0] if row else None
        finally:
            conn.close()
    except Exception as e:
        print(f"⚠️ Кэш не прочитался ({table}): {e}")
        return None


def _cache_put(table: str, col: str, key: str, payload) -> None:
    try:
        conn = _conn()
        if not conn:
            return
        try:
            from psycopg2.extras import Json
            with conn, conn.cursor() as cur:
                _ensure_tables(cur)
                cur.execute(
                    f"INSERT INTO {table} ({col}, payload) VALUES (%s, %s) "
                    f"ON CONFLICT ({col}) DO UPDATE SET payload = EXCLUDED.payload",
                    (key, Json(payload)),
                )
        finally:
            conn.close()
    except Exception as e:
        print(f"⚠️ Кэш не записался ({table}): {e}")


# ── Город → координаты и часовой пояс ────────────────────────────────
def _geo_fetch(q: str, lang: str) -> list[dict]:
    import urllib.request
    import urllib.parse
    url = GEO_API + "?" + urllib.parse.urlencode(
        {"name": q, "count": 8, "language": lang, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": GEO_UA,
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=GEO_TIMEOUT) as r:
        data = json.loads(r.read(400_000).decode("utf-8"))
    out = []
    for it in data.get("results") or []:
        if not it.get("timezone"):
            continue                    # без пояса город бесполезен: время не перевести
        out.append({
            "name": it.get("name", ""),
            "country": it.get("country", ""),
            "region": it.get("admin1") or "",
            "lat": round(float(it["latitude"]), 4),
            "lon": round(float(it["longitude"]), 4),
            "tz": it["timezone"],
            "pop": int(it.get("population") or 0),
        })
    # крупные города выше: человек, набравший «Москва», имеет в виду ту самую
    out.sort(key=lambda c: -c["pop"])
    return out[:6]


def geo_search(q: str) -> list[dict]:
    q = (q or "").strip()[:64]
    if len(q) < 2:
        return []
    # Кириллица — ищем по-русски, латиница — по-английски: провайдер отдаёт
    # названия на том языке, на котором спросили, и так они и лягут в карту.
    lang = "ru" if re.search(r"[Ѐ-ӿ]", q) else "en"
    key = f"{lang}:{q.lower()}"
    hit = _cache_get("geo_cache", "q", key)
    if hit is not None:
        return hit
    try:
        res = _geo_fetch(q, lang)
    except Exception as e:
        print(f"⚠️ Геокодер недоступен: {e}")
        return []
    _cache_put("geo_cache", "q", key, res)
    return res


# ── Факты для толкования ─────────────────────────────────────────────
# Планеты, которые идут по зодиаку годами: их взаимные углы одинаковы у всех,
# кто родился в один сезон. Называть это чертой характера — враньё,
# и мы прямо помечаем такие аспекты как поколенческие.
SLOW = {"Юпитер", "Сатурн", "Уран", "Нептун", "Плутон"}

HOUSE_MEANS = {
    1: "внешность и первое впечатление", 2: "деньги и вещи", 3: "речь, учёба, ближний круг",
    4: "дом и семья", 5: "игра, дети, влюблённость", 6: "работа и режим",
    7: "партнёрство", 8: "чужие ресурсы и кризисы", 9: "дальние поездки и убеждения",
    10: "профессия и репутация", 11: "друзья и планы", 12: "уединение и то, что скрыто",
}


def _arcmin(deg: float) -> str:
    m = deg * 60
    if m < 1:
        return f"{m * 60:.0f}″"
    if deg < 1:
        return f"{m:.0f}′"
    return f"{deg:.2f}°"


def digest(ch: dict) -> dict:
    """Сухая выжимка карты: только то, что посчитано, без интерпретаций."""
    by = {p["name"]: p for p in ch["planets"]}
    lines = []
    for p in ch["planets"]:
        lines.append(f"{p['name']}: {p['label']}, дом {p['house']}"
                     f"{' , ретроградна' if p['retro'] else ''}")
    asp = []
    for a in ch["aspects"][:12]:
        gen = a["a"] in SLOW and a["b"] in SLOW
        asp.append({
            "text": f"{a['a']} {a['type']} {a['b']}, расхождение {_arcmin(a['exact'])}",
            "generational": gen,
            "exact": a["exact"],
        })
    tight = [a for a in ch["aspects"] if a["exact"] <= 1.0]
    retro = [p["name"] for p in ch["planets"] if p["retro"]]
    return {
        "asc": ch["asc"]["label"],
        "mc": ch["mc"]["label"],
        "sun": by["Солнце"], "moon": by["Луна"],
        "positions": lines,
        "aspects": asp,
        "tight": [f"{a['a']} {a['type']} {a['b']} ({_arcmin(a['exact'])})" for a in tight],
        "retro": retro,
        "houses": HOUSE_MEANS,
    }


SYSTEM_NATAL = (
    "Ты пишешь толкование натальной карты для страницы «Эфемерида».\n"
    "Голос: сухой, точный, без восторга и без мистики. Так говорит человек, "
    "который умеет считать и не путает расчёт с выводом.\n"
    "\n"
    "Правила, нарушать которые нельзя:\n"
    "1. Все числа, знаки, дома и аспекты берутся ТОЛЬКО из выданных фактов. "
    "Ничего не пересчитывай и не добавляй от себя.\n"
    "2. Аспект, помеченный как поколенческий, объясняй как отпечаток эпохи, "
    "а не как черту человека, и говори об этом прямо.\n"
    "3. Никаких предсказаний, диагнозов, советов про здоровье, деньги и отношения. "
    "Никакой лести и никаких «вы особенный».\n"
    "4. Астрономию подавай как факт, характер — как толкование, и не смешивай их. "
    "Где уместно, показывай масштаб: сравнивай угол с видимым диском Луны (полградуса).\n"
    "5. Пиши по-русски, живо, без канцелярита и без эзотерического словаря "
    "(«энергии», «вибрации», «кармические задачи» — запрещены).\n"
    "\n"
    "Разметка: **жирный** для первой фразы абзаца. Ничего другого не используй.\n"
    "\n"
    "Формат ответа — строго JSON без пояснений:\n"
    '{"lead": "одно предложение: асцендент, MC, Солнце, Луна", '
    '"blocks": ["абзац", "абзац", "абзац", "абзац"], '
    '"verdict": "две короткие строки через \\n: степень опасности и рекомендация"}\n'
    "Абзацев четыре или пять, каждый — про один заметный факт карты."
)


def _fallback_reading(d: dict) -> dict:
    """Если модель недоступна — отдаём честный сухой разбор без неё."""
    blocks = [
        "**Карта посчитана, толкование — нет.** Модель сейчас недоступна, "
        "поэтому ниже только то, что даёт астрономия: положения, дома и углы. "
        "Это и есть та часть, которую можно проверить.",
    ]
    if d["tight"]:
        blocks.append("**Самые точные углы карты:** " + "; ".join(d["tight"][:3]) + ".")
    if d["retro"]:
        blocks.append("**Ретроградны:** " + ", ".join(d["retro"]) +
                      ". Это видимое движение назад по небу, а не свойство характера.")
    return {
        "lead": f"Асцендент — {d['asc']}. Середина неба — {d['mc']}. "
                f"Солнце в знаке {d['sun']['sign']}, Луна в знаке {d['moon']['sign']}.",
        "blocks": blocks,
        "verdict": "Степень опасности: нулевая.\nРекомендация: зайти позже за словами.",
    }


def interpret(ch: dict, ask) -> dict:
    d = digest(ch)
    facts = {
        "асцендент": d["asc"], "середина неба": d["mc"],
        "положения": d["positions"],
        "аспекты": [
            (a["text"] + (" — ПОКОЛЕНЧЕСКИЙ" if a["generational"] else ""))
            for a in d["aspects"]
        ],
        "особо точные": d["tight"],
        "ретроградные": d["retro"],
        "значения домов": d["houses"],
    }
    prompt = ("Факты карты:\n" + json.dumps(facts, ensure_ascii=False, indent=1) +
              "\n\nНапиши толкование по правилам. Только JSON.")
    try:
        raw = ask(prompt, system=SYSTEM_NATAL, temperature=0.9) or ""
    except Exception as e:
        print(f"⚠️ Толкование не сгенерировалось: {e}")
        return _fallback_reading(d)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return _fallback_reading(d)
    try:
        out = json.loads(m.group(0))
    except Exception:
        return _fallback_reading(d)
    lead = str(out.get("lead") or "")[:400]
    blocks = [str(b)[:1200] for b in (out.get("blocks") or [])][:6]
    verdict = str(out.get("verdict") or "")[:300]
    if not blocks:
        return _fallback_reading(d)
    return {"lead": lead, "blocks": blocks, "verdict": verdict}


# ── Ручки ────────────────────────────────────────────────────────────
class ChartIn(BaseModel):
    date: str = ""        # 1990-05-17
    time: str = "12:00"   # 10:50
    lat: float = 0.0
    lon: float = 0.0
    tz: str = ""
    place: str = ""       # как показать место в шапке
    unknown_time: bool = False


def _parse_when(body: ChartIn) -> tuple[dt.datetime, str] | tuple[None, str]:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", body.date or ""):
        return None, "Дата нужна в виде 1990-05-17."
    hhmm = body.time or "12:00"
    if not re.fullmatch(r"\d{1,2}:\d{2}", hhmm):
        return None, "Время нужно в виде 10:50."
    try:
        y, mo, da = (int(x) for x in body.date.split("-"))
        hh, mi = (int(x) for x in hhmm.split(":"))
        when = dt.datetime(y, mo, da, hh, mi)
    except ValueError:
        return None, "Такой даты не существует."
    if not (MIN_YEAR <= y <= dt.date.today().year):
        return None, f"Год должен быть между {MIN_YEAR} и нынешним."
    return when, ""


def build_router(ask, rate_ok, client_ip) -> APIRouter:
    router = APIRouter(prefix="/natal", tags=["natal"])

    @router.get("/suggest")
    def suggest(q: str = "", request: Request = None):
        if request is not None and not rate_ok(client_ip(request), 60):
            return {"cities": [], "error": "Слишком часто. Подождите минуту."}
        return {"cities": geo_search(q)}

    @router.post("/chart")
    def chart_endpoint(body: ChartIn, request: Request):
        if engine is None:
            return {"error": "Расчёт временно недоступен: " + ENGINE_ERROR}
        if not rate_ok(client_ip(request), 20):
            return {"error": "Слишком много запросов подряд. Попробуйте через несколько минут."}

        when, err = _parse_when(body)
        if err:
            return {"error": err}
        tz = (body.tz or "").strip()
        try:
            ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError):
            return {"error": "Не понял часовой пояс. Выберите город из подсказки."}
        if not (-90 <= body.lat <= 90) or not (-180 <= body.lon <= 180):
            return {"error": "Координаты вне Земли."}

        key = (f"{body.date}|{body.time}|{round(body.lat, 3)}|{round(body.lon, 3)}"
               f"|{tz}|{int(bool(body.unknown_time))}")
        cached = _cache_get("natal_readings", "key", key)
        if cached:
            return cached

        try:
            ch = engine.chart(when, tz, body.lat, body.lon)
        except Exception as e:
            print(f"⚠️ Карта не посчиталась: {e}")
            return {"error": "Расчёт не сошёлся. Проверьте дату и место."}

        reading = interpret(ch, ask)
        # Время неизвестно — дома и углы карты бессмысленны: они проворачиваются
        # на весь круг за сутки. Честнее сказать это, чем молча показать.
        payload = {
            "chart": ch,
            "reading": reading,
            "place": (body.place or "").strip()[:80],
            "tz": tz,
            "when": f"{body.date} {body.time}",
            "utc_offset": dt.datetime(when.year, when.month, when.day, when.hour,
                                      when.minute, tzinfo=ZoneInfo(tz)).utcoffset().total_seconds() / 3600,
            "time_known": not body.unknown_time,
            "note": ("" if not body.unknown_time else
                     "Время рождения не указано — взят полдень. Дома, асцендент и "
                     "середина неба при этом недостоверны: за сутки они делают полный оборот. "
                     "Положения планет в знаках верны, кроме Луны — она за сутки проходит "
                     "до 15 градусов."),
        }
        _cache_put("natal_readings", "key", key, payload)
        return payload

    return router
