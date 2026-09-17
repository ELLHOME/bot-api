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

# Версия толкования. Меняем её, когда правим промпт: старые разборы в кэше
# написаны прежним голосом, и отдавать их вперемешку с новыми нечестно.
# Строки с прошлой версией просто перестают находиться.
READING_V = 3

# Версия раздела «что сейчас». Он живёт своим кэшем: карта рождения
# не меняется никогда, а небо над ней — каждый день.
TRANSIT_V = 1

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
    cur.execute("CREATE TABLE IF NOT EXISTS transit_readings ("
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
        {"name": q, "count": 20, "language": lang, "format": "json"})
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
    return out


def rank_places(places: list[dict], q: str, limit: int = 7) -> list[dict]:
    """Сортировка подсказок.

    По населению сортировать нельзя: деревню, которая называется как
    известный город, оно утопит ниже списка, и человек решит, что его
    места в базе нет. Сначала точное совпадение названия, потом начало
    слова, и только внутри каждой группы — по населению.
    """
    ql = (q or "").strip().lower()

    def rank(c):
        n = (c.get("name") or "").strip().lower()
        grade = 0 if n == ql else 1 if n.startswith(ql) else 2
        return (grade, -int(c.get("pop") or 0))

    return sorted(places, key=rank)[:limit]


# Слова, которые есть у половины регионов и потому ничего не уточняют.
# Без этого «Красноярская обл» цеплялась бы за любую «область» в стране.
GENERIC = {"обл", "область", "области", "край", "края", "район", "районе",
           "респ", "республика", "республике", "ао", "округ", "округе",
           "губерния", "oblast", "region", "province", "district", "county", "state"}


def _matches_region(c: dict, words: list[str]) -> bool:
    """Слова после названия — уточнение места: регион или страна.

    Сравниваем по общему началу, а не по точному совпадению: человек пишет
    «Красноярская обл», а в базе «Красноярский край», и оба должны сойтись.
    Хватает одного попавшего слова — «обл» и «р-н» ни с чем не совпадут,
    и требовать от них попадания значит наказывать за вежливость.
    """
    hay = f"{c.get('region', '')} {c.get('country', '')}".lower()
    parts = [w for w in re.split(r"[\s,.\-]+", hay) if len(w) > 2 and w not in GENERIC]
    words = [w for w in words if w not in GENERIC]
    if not words:                 # уточнили одними «обл» и «край» — значит, не уточнили
        return True

    def close(a: str, b: str) -> bool:
        need = min(4, len(a), len(b))
        return need >= 3 and a[:need] == b[:need]

    return any(close(w, p) for w in words if len(w) > 2 for p in parts)


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
        res = rank_places(_geo_fetch(q, lang), q)
        # Одноимённых деревень в стране бывают десятки, а мест в подсказке
        # семь. Поэтому если целиком запрос ничего не дал, разбираем его как
        # «название + уточнение»: ищем по первому слову, остальным фильтруем
        # регион. Полный запрос пробуем первым — многословные названия
        # вроде «Нижний Новгород» должны находиться как есть.
        words = q.split()
        if len(words) > 1:
            head, tail = words[0], [w.lower() for w in words[1:]]
            if len(head) >= 2:
                wide = rank_places(_geo_fetch(head, lang), head, limit=40)
                narrow = [c for c in wide if _matches_region(c, tail)]
                if narrow:
                    res = narrow[:7]
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

# Знак в предложном падеже: «Солнце во Льве», а не «Солнце в Лев».
# Лид собирается здесь, а не моделью: это голая справка, и пусть она
# будет одинаковой всегда.
IN_SIGN = {
    "Овен": "в Овне", "Телец": "в Тельце", "Близнецы": "в Близнецах",
    "Рак": "в Раке", "Лев": "во Льве", "Дева": "в Деве",
    "Весы": "в Весах", "Скорпион": "в Скорпионе", "Стрелец": "в Стрельце",
    "Козерог": "в Козероге", "Водолей": "в Водолее", "Рыбы": "в Рыбах",
}

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

    ph = ch.get("moon_phase") or {}
    rl = ch.get("ruler") or {}
    ang = [f"{a['a']} {a['type']} {a['b']}, расхождение {_arcmin(a['exact'])}"
           for a in (ch.get("angle_aspects") or [])[:8]]
    el = ch.get("elements") or {}
    missing = [e for e in ("огонь", "земля", "воздух", "вода") if not el.get(e)]

    return {
        "asc": ch["asc"]["label"],
        "mc": ch["mc"]["label"],
        "sun": by["Солнце"], "moon": by["Луна"],
        "positions": lines,
        "aspects": asp,
        "angle_aspects": ang,
        "tight": [f"{a['a']} {a['type']} {a['b']} ({_arcmin(a['exact'])})" for a in tight],
        "retro": retro,
        "houses": HOUSE_MEANS,
        "moon_phase": (f"{ph.get('name','')}, освещено {ph.get('illum','?')}% диска, "
                       f"Луна отошла от Солнца на {ph.get('angle','?')}°") if ph else "",
        "sector": ("дневная карта: Солнце над горизонтом"
                   if ch.get("day_chart") else "ночная карта: Солнце под горизонтом"),
        "ruler": (f"управитель асцендента — {rl.get('planet','')}"
                  + (f" (в старой традиции {rl['classic']})" if rl.get("classic") else "")
                  + (f", стоит {rl.get('label','')} в доме {rl.get('house','')}"
                     if rl.get("label") else "")) if rl else "",
        "elements": el,
        "elements_missing": missing,
        "modes": ch.get("modes") or {},
        "stelliums": [f"{g['where']}: {', '.join(g['who'])}"
                      for g in (ch.get("stelliums") or [])],
        "stations": ch.get("stations") or [],
        "lilith": (ch.get("lilith") or {}).get("label", ""),
    }


VOICE = (
    "ГОЛОС. Сухой, точный, с холодноватой усмешкой. Так пишет человек, который "
    "умеет считать и не путает расчёт с выводом. Короткие предложения. "
    "Порядок внутри абзаца: сначала факт, потом что он значит по ремеслу, "
    "потом — по-житейски, обычными словами, как сказал бы приятель.\n"
    "\n"
    "ЗАПРЕЩЕНЫ обороты, которыми пишут справки и гороскопы. Ни одного из них: "
    "«свидетельствует», «проявляется», «обозначает», «указывает на», «является», "
    "«характеризуется», «отражает», «определяет», «формирует», «на уровне характера», "
    "«в толковании это», «в поведении это», «данный», «базовая структура», "
    "«структура личности», «астрономическое положение», «энергии», «вибрации», "
    "«кармические задачи», «личностные планы». Отглагольных существительных "
    "(«восприятие», «самовыражение», «социализация») — не больше одного на абзац.\n"
)

SYSTEM_NOW = (
    "Ты пишешь раздел «что сейчас» для страницы «Эфемерида».\n"
    "\n" + VOICE + "\n"
    "Тебе дают сегодняшнее небо и углы, которые сегодняшние планеты образуют "
    "к точкам карты рождения. Орбы и даты посчитаны, твоё дело — рассказать.\n"
    "\n"
    "ПРАВИЛА, нарушать которые нельзя:\n"
    "1. Все числа, названия и даты — только из выданных фактов. Ничего не считай сам.\n"
    "2. НИКАКИХ предсказаний и НИКАКИХ советов. Не пиши, что человеку делать, чего "
    "избегать, что его ждёт, когда начинать и когда воздержаться. Ты описываешь "
    "конфигурацию неба и то, КАК ЕЁ ПРИНЯТО ТОЛКОВАТЬ. Это разные вещи, и разница "
    "должна быть видна в тексте: «так принято читать», «астрологи называют это», "
    "«традиция приписывает».\n"
    "3. Честно про срок. Солнце, Меркурий, Венера и Марс проходят точку за дни — "
    "это короткий эпизод. Юпитер, Сатурн, Уран, Нептун и Плутон стоят месяцами, "
    "а с учётом ретроградных петель возвращаются по нескольку раз.\n"
    "4. Аспект «расходится» — он уже был точен, это позади. «Сходится» — назови дату, "
    "когда угол станет точным. «Точен сейчас» — так и скажи.\n"
    "5. Не льсти и не пугай. Ни одного «вас ждёт», ни одного «будьте осторожны».\n"
    "\n"
    "Разметка: **жирный** для первой фразы абзаца, больше ничего.\n"
    "\n"
    "Формат ответа — строго JSON без пояснений:\n"
    '{"blocks": ["абзац", "абзац", "абзац"], "tail": "одна строка"}\n'
    "Абзацев два или три, про самые точные углы, и каждый про свой. "
    "tail — одна сухая строка о том, что всё перечисленное относится к небу, "
    "а не к жизни, и совпадения человек проверяет сам."
)

SYSTEM_NATAL = (
    "Ты пишешь разбор натальной карты для страницы «Эфемерида».\n"
    "\n"
    + VOICE +
    "\n"
    "Вот образец голоса. Карта чужая, факты оттуда брать нельзя — нужен только тон:\n"
    "\n"
    "«**Венера и Марс стоят в одной точке.** Расхождение — девять угловых минут, "
    "втрое меньше видимого диска Луны. И стоят они прямо на асценденте: на той "
    "точке горизонта, что поднималась в момент рождения. По ремеслу это значит, "
    "что желание и усилие запущены одним механизмом. По-житейски: нравиться и "
    "добиваться — одно действие, и человек искренне не понимает, как их разделить.»\n"
    "\n"
    "«**Луна в Козероге в четвёртом доме.** Дом отвечает за дом — в смысле жилища "
    "и семьи. Козерог означает, что чувства предпочитают иметь смету. Не холодность, "
    "а недоверие к словам без подтверждения.»\n"
    "\n"
    "«**Сатурн в точном квадрате к Плутону.** Прямой угол выдержан до одной угловой "
    "минуты — такое не разглядеть невооружённым глазом. Соблазнительно сделать отсюда "
    "вывод о характере, но честнее сказать иначе: летом 2010-го это стояло у каждого "
    "новорождённого на планете. Личного тут ноль, это отпечаток эпохи.»\n"
    "\n"
    "ПРАВИЛА, нарушать которые нельзя:\n"
    "1. Все числа, знаки, дома и аспекты берутся ТОЛЬКО из выданных фактов. "
    "Ничего не пересчитывай и не добавляй от себя.\n"
    "2. Аспект, помеченный как ПОКОЛЕНЧЕСКИЙ, объясняй как отпечаток эпохи, "
    "а не как черту человека, и говори об этом прямо, как в третьем образце.\n"
    "3. Никаких предсказаний, диагнозов, советов про здоровье, деньги и отношения. "
    "Никакой лести и никаких «вы особенный».\n"
    "4. Астрономию подавай как факт, характер — как толкование, и не смешивай их. "
    "Где уместно, показывай масштаб: сравнивай угол с видимым диском Луны (полградуса).\n"
    "5. Каждый абзац — про один заметный факт карты, и начинается он с этого факта, "
    "выделенного **жирным**. Больше никакой разметки.\n"
    "\n"
    "Формат ответа — строго JSON без пояснений:\n"
    '{"blocks": ["абзац", "абзац", "абзац", "абзац", "абзац", "абзац"], '
    '"verdict": "две строки через \\n"}\n'
    "Абзацев шесть или семь, и они не должны быть про одно и то же: возьми разные "
    "стороны карты — точный аспект, положение в доме, аспект к асценденту или MC, "
    "фазу Луны, перекос по стихиям или пустую стихию, скопление планет в одном знаке, "
    "планету на станции. Вердикт — ровно две короткие строки: "
    "«Степень опасности: <одно слово>.» и «Рекомендация: <полстроки>.» "
    "Рекомендация не про жизнь, а про обращение с этим текстом."
)


def _lead(ch: dict) -> str:
    by = {p["name"]: p for p in ch["planets"]}
    return (f'Асцендент — {ch["asc"]["label"]}. '
            f'Середина неба — {ch["mc"]["label"]}. '
            f'Солнце {IN_SIGN.get(by["Солнце"]["sign"], "в " + by["Солнце"]["sign"])}, '
            f'Луна {IN_SIGN.get(by["Луна"]["sign"], "в " + by["Луна"]["sign"])}.')


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
        "lead": "",
        "blocks": blocks,
        "verdict": "Степень опасности: нулевая.\nРекомендация: зайти позже за словами.",
    }


def interpret(ch: dict, ask) -> dict:
    d = digest(ch)
    lead = _lead(ch)
    facts = {
        "асцендент": d["asc"], "середина неба": d["mc"],
        "положения": d["positions"],
        "аспекты": [
            (a["text"] + (" — ПОКОЛЕНЧЕСКИЙ" if a["generational"] else ""))
            for a in d["aspects"]
        ],
        "аспекты к углам карты": d["angle_aspects"],
        "особо точные": d["tight"],
        "ретроградные": d["retro"],
        "на станции (почти стоят, скоро развернутся)": d["stations"],
        "фаза Луны": d["moon_phase"],
        "время суток": d["sector"],
        "управитель": d["ruler"],
        "стихии": d["elements"],
        "стихии без единого тела": d["elements_missing"],
        "кресты": d["modes"],
        "скопления": d["stelliums"],
        "чёрная луна (апогей орбиты Луны)": d["lilith"],
        "значения домов": d["houses"],
    }
    prompt = ("Факты карты:\n" + json.dumps(facts, ensure_ascii=False, indent=1) +
              "\n\nНапиши толкование по правилам. Только JSON.")
    try:
        raw = ask(prompt, system=SYSTEM_NATAL, temperature=0.9) or ""
    except Exception as e:
        print(f"⚠️ Толкование не сгенерировалось: {e}")
        return {**_fallback_reading(d), "lead": lead}
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {**_fallback_reading(d), "lead": lead}
    try:
        out = json.loads(m.group(0))
    except Exception:
        return {**_fallback_reading(d), "lead": lead}
    blocks = [str(b)[:1200] for b in (out.get("blocks") or [])][:8]
    verdict = str(out.get("verdict") or "")[:300]
    if not blocks:
        return {**_fallback_reading(d), "lead": lead}
    return {"lead": lead, "blocks": blocks, "verdict": verdict}


def _ru_date(iso: str) -> str:
    MONTHS = ["января","февраля","марта","апреля","мая","июня",
              "июля","августа","сентября","октября","ноября","декабря"]
    d = dt.date.fromisoformat(iso)
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def now_facts(tr: dict) -> dict:
    """Сегодняшнее небо и попадания в карту — в виде, понятном модели."""
    sky = [f"{p['name']}: {p['label']}" + (", ретроградно" if p["retro"] else "")
           for p in tr["sky"]]
    m = tr["moon"]
    hits = []
    for h in tr["hits"][:8]:
        when = ("точен сегодня" if h["state"] == "точен сейчас"
                else f"{h['state']}, точный угол {_ru_date(h['exact'])}")
        hits.append(f"{h['who']}{' (ретроградно)' if h['retro'] else ''} "
                    f"{h['type']} к натальному {h['to']}: "
                    f"сейчас {_arcmin(h['orb'])} от точного, {when}, "
                    f"{'медленная планета — эпизод на месяцы' if h['slow'] else 'быстрая планета — эпизод на дни'}")
    return {
        "дата": _ru_date(tr["when"]),
        "небо сегодня": sky,
        "Луна сегодня": f"{m['label']}, {m['phase']}, освещено {m['illum']}%",
        "углы к карте рождения": hits,
    }


def interpret_now(tr: dict, ask) -> dict:
    prompt = ("Факты:\n" + json.dumps(now_facts(tr), ensure_ascii=False, indent=1) +
              "\n\nНапиши раздел «что сейчас» по правилам. Только JSON.")
    try:
        raw = ask(prompt, system=SYSTEM_NOW, temperature=0.9) or ""
        m = re.search(r"\{.*\}", raw, re.S)
        out = json.loads(m.group(0)) if m else {}
    except Exception as e:
        print(f"⚠️ Раздел «что сейчас» не сгенерировался: {e}")
        out = {}
    blocks = [str(b)[:1200] for b in (out.get("blocks") or [])][:4]
    if not blocks:
        top = tr["hits"][0] if tr["hits"] else None
        blocks = ["**Небо посчитано, слова — нет.** Модель сейчас недоступна, "
                  "поэтому ниже только таблица: где планеты стоят сегодня и какие "
                  "углы они держат к карте рождения."]
        if top:
            blocks.append(f"**Самый точный угол на сегодня** — {top['who']} "
                          f"{top['type']} к натальному {top['to']}, "
                          f"{_arcmin(top['orb'])} от точного.")
    return {"blocks": blocks,
            "tail": str(out.get("tail") or
                        "Всё перечисленное — про небо, а не про жизнь. "
                        "Совпадения проверяйте сами.")[:300]}


# ── Часы, которых не бывает, и часы, которые бывают дважды ───────────
# Дважды в год перевод стрелок ломает однозначность «местного времени».
# Осенью час повторяется, весной час пропадает. Ошибка ровно в час —
# для планет это мелочь, а асцендент уезжает на пол-знака. Молчать об этом
# на странице, которая называется «посчитанная всерьёз», нельзя.


def _fmt_off(a: dt.datetime) -> str:
    h = a.utcoffset().total_seconds() / 3600
    return f"UTC{'+' if h >= 0 else '−'}{abs(h):g}"


def tz_trouble(naive: dt.datetime, tz: str) -> tuple[str, str]:
    """Возвращает (вид, подпись). Вид: '' | 'ambiguous' | 'skipped'."""
    zone = ZoneInfo(tz)
    a0 = naive.replace(tzinfo=zone, fold=0)
    a1 = naive.replace(tzinfo=zone, fold=1)
    if a0.utcoffset() == a1.utcoffset():
        return "", ""
    # Пропущенный час узнаём по тому, что оно не переживает оборот в UTC и обратно:
    # такого показания на часах в этой зоне просто не было.
    back = a0.astimezone(dt.timezone.utc).astimezone(zone).replace(tzinfo=None)
    kind = "ambiguous" if back == naive else "skipped"
    return kind, f"{_fmt_off(a0)} / {_fmt_off(a1)}"


def tz_note(naive: dt.datetime, tz: str, lat: float, lon: float) -> str:
    kind, _ = tz_trouble(naive, tz)
    if not kind:
        return ""
    zone = ZoneInfo(tz)
    a0, a1 = naive.replace(tzinfo=zone, fold=0), naive.replace(tzinfo=zone, fold=1)
    if kind == "skipped":
        return (f"В эту ночь стрелки переводили вперёд, и такого времени на часах "
                f"не существовало — час был пропущен. Карта посчитана по {_fmt_off(a0)}, "
                f"как если бы время записали по старым стрелкам. Если точность важна, "
                f"стоит уточнить час рождения.")
    # Неоднозначный час: показываем, чем именно отличался бы второй проход.
    # Это дешевле объяснений — видно своими глазами.
    try:
        alt = engine.chart(naive, tz, lat, lon, fold=1)
        tail = (f" Во втором случае асцендент был бы {alt['asc']['label']} "
                f"вместо того, что в шапке, а дома сдвинулись бы вместе с ним. "
                f"Положения планет в знаках при этом почти те же.")
    except Exception:
        tail = " Во втором случае вся карта сдвинулась бы примерно на пол-знака."
    return (f"В эту ночь стрелки переводили назад, и такое время было на часах дважды: "
            f"сначала по {_fmt_off(a0)}, через час — по {_fmt_off(a1)}. "
            f"Взят первый проход.{tail}")


SYSTEM_ASK = (
    "Ты отвечаешь на вопрос человека на странице «Эфемерида». У тебя есть его "
    "карта рождения и сегодняшнее небо над ней.\n"
    "\n" + VOICE + "\n"
    "ЧТО ТЫ ДЕЛАЕШЬ. Человек пришёл с вопросом — «что делать», «чем кончится», "
    "«стоит ли». Ты не знаешь ответа, и никакая карта его не знает. Но ты умеешь "
    "две вещи: назвать точный факт про небо и задать вопрос, который человек "
    "сам себе не задал. Этого обычно хватает.\n"
    "\n"
    "ЗАПРЕЩЕНО НАСТРОГО:\n"
    "1. Предсказывать. Никаких «вас ждёт», «в октябре начнётся», «этот человек "
    "вернётся». Ты не знаешь будущего.\n"
    "2. Указывать, что делать. Никаких «увольняйтесь», «подождите до ноября», "
    "«не подписывайте». Решение принимает он, и последствия несёт тоже он.\n"
    "3. Советовать по здоровью, деньгам, лекарствам и юридическим делам. "
    "Тут отправляй к тем, кто в этом разбирается, — врачу, юристу.\n"
    "4. Делать вид, что расположение планет определяет исход. Астрология "
    "в этом разговоре — повод подумать, а не причина событий. Держи это видимым.\n"
    "5. Льстить, пугать и набивать цену. Никакой мистики и никакого «вы особенный».\n"
    "\n"
    "ЕСЛИ В ВОПРОСЕ БЕДА. Насилие, мысли о смерти, тяжёлая болезнь, отчаяние — "
    "брось астрологию совсем. Не считай, не толкуй, не шути. Скажи прямо и коротко, "
    "по-человечески: карта тут ничем не поможет, и с таким стоит идти к живому "
    "человеку — тому, кому доверяешь, или к специалисту. Три-четыре предложения, "
    "без глифов и без градусов.\n"
    "\n"
    "КАК ОТВЕЧАТЬ В ОБЫЧНОМ СЛУЧАЕ. Два-три коротких абзаца, не больше.\n"
    "— Возьми из фактов то, что правда относится к вопросу. Если ничего "
    "подходящего в небе нет, так и скажи: сегодня по этой части тихо. Это "
    "честнее, чем притягивать за уши.\n"
    "— Числа и даты — только выданные, своих не придумывай.\n"
    "— Закончи вопросом к человеку. Не риторическим, а таким, на который он "
    "может ответить себе сам и сдвинуться с места.\n"
    "\n"
    "Разметка: **жирный** для одной ключевой фразы, не больше. Никакого JSON, "
    "пиши обычным текстом, абзацы разделяй пустой строкой."
)


def answer_question(ch: dict, tr: dict | None, question: str, ask) -> str:
    facts = {
        "карта рождения": {
            "асцендент": ch["asc"]["label"], "середина неба": ch["mc"]["label"],
            "положения": [f"{p['name']}: {p['label']}, дом {p['house']}"
                          f"{', ретроградна' if p['retro'] else ''}" for p in ch["planets"]],
            "самые точные аспекты": [
                f"{a['a']} {a['type']} {a['b']} ({_arcmin(a['exact'])})"
                for a in ch["aspects"][:6]],
        },
    }
    if tr:
        facts["сегодня"] = now_facts(tr)
    prompt = ("Факты:\n" + json.dumps(facts, ensure_ascii=False, indent=1) +
              "\n\nВопрос человека: " + question.strip() +
              "\n\nОтветь по правилам.")
    try:
        out = (ask(prompt, system=SYSTEM_ASK, temperature=0.9) or "").strip()
    except Exception as e:
        print(f"⚠️ Ответ на вопрос не сгенерировался: {e}")
        return ""
    # Модель иногда сползает в JSON, хотя просили текст.
    if out.startswith("{"):
        try:
            d = json.loads(out)
            out = " ".join(str(v) for v in d.values() if isinstance(v, str))
        except Exception:
            pass
    return out[:2500]


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


class AskIn(ChartIn):
    question: str = ""


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

        key = (f"v{READING_V}|{body.date}|{body.time}|{round(body.lat, 3)}"
               f"|{round(body.lon, 3)}|{tz}|{int(bool(body.unknown_time))}")
        # Предупреждение про перевод стрелок считается заново на каждый ответ:
        # оно выводится из тех же данных и стоит доли миллисекунды, зато старые
        # строки в кэше не надо пересчитывать ради нового поля.
        warn = tz_note(when, tz, body.lat, body.lon) if engine else ""

        cached = _cache_get("natal_readings", "key", key)

        try:
            ch = engine.chart(when, tz, body.lat, body.lon)
        except Exception as e:
            print(f"⚠️ Карта не посчиталась: {e}")
            return {"error": "Расчёт не сошёлся. Проверьте дату и место."}

        # Небо над картой меняется каждый день, сама карта — никогда.
        # Поэтому у раздела «что сейчас» свой кэш, со сроком в сутки,
        # и считается он параллельно с разбором: оба запроса сетевые,
        # ждать их по очереди значит удвоить паузу на первом заходе.
        try:
            tr = engine.transits(ch)
        except Exception as e:
            print(f"⚠️ Транзиты не посчитались: {e}")
            tr = None

        now_key = f"t{TRANSIT_V}|{key}|{tr['when']}" if tr else ""
        now_cached = _cache_get("transit_readings", "key", now_key) if tr else None

        if cached is not None:
            reading = cached.get("reading") or {}
        if now_cached is not None and cached is not None:
            now_text = now_cached
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_read = pool.submit(interpret, ch, ask) if cached is None else None
                f_now = (pool.submit(interpret_now, tr, ask)
                         if tr and now_cached is None else None)
                reading = f_read.result() if f_read else (cached or {}).get("reading") or {}
                now_text = f_now.result() if f_now else now_cached
            if tr and now_cached is None and now_text:
                _cache_put("transit_readings", "key", now_key, now_text)
        # Время неизвестно — дома и углы карты бессмысленны: они проворачиваются
        # на весь круг за сутки. Честнее сказать это, чем молча показать.
        payload = {
            "chart": ch,
            "reading": reading,
            "now": ({**tr, "text": now_text} if tr and now_text else None),
            "place": (body.place or "").strip()[:80],
            "tz": tz,
            "when": f"{body.date} {body.time}",
            "utc_offset": dt.datetime(when.year, when.month, when.day, when.hour,
                                      when.minute, tzinfo=ZoneInfo(tz)).utcoffset().total_seconds() / 3600,
            "time_known": not body.unknown_time,
            "tz_note": warn,
            "note": ("" if not body.unknown_time else
                     "Время рождения не указано — взят полдень. Дома, асцендент и "
                     "середина неба при этом недостоверны: за сутки они делают полный оборот. "
                     "Положения планет в знаках верны, кроме Луны — она за сутки проходит "
                     "до 15 градусов."),
        }
        # В кэш карты кладём только то, что не зависит от сегодняшнего дня.
        if cached is None:
            _cache_put("natal_readings", "key", key,
                       {k: v for k, v in payload.items() if k not in ("now", "tz_note")})
        return payload

    @router.post("/ask")
    def ask_endpoint(body: AskIn, request: Request):
        if engine is None:
            return {"error": "Расчёт временно недоступен."}
        q = (body.question or "").strip()[:400]
        if len(q) < 3:
            return {"error": "Спросите что-нибудь."}
        # Вопросы не кэшируются — каждый свой. Поэтому лимит строже,
        # чем на расчёт карты: это единственное, что стоит денег на каждый заход.
        if not rate_ok(client_ip(request), 8):
            return {"error": "Слишком много вопросов подряд. Вернитесь через несколько минут."}

        when, err = _parse_when(body)
        if err:
            return {"error": err}
        tz = (body.tz or "").strip()
        try:
            ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError):
            return {"error": "Не понял часовой пояс."}
        if not (-90 <= body.lat <= 90) or not (-180 <= body.lon <= 180):
            return {"error": "Координаты вне Земли."}

        try:
            ch = engine.chart(when, tz, body.lat, body.lon)
        except Exception as e:
            print(f"⚠️ Карта не посчиталась: {e}")
            return {"error": "Расчёт не сошёлся."}
        try:
            tr = engine.transits(ch)
        except Exception:
            tr = None

        text = answer_question(ch, tr, q, ask)
        if not text:
            return {"error": "Ответ не сложился. Попробуйте ещё раз."}
        return {"question": q, "answer": text}

    return router
