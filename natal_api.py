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
READING_V = 6

# Версия раздела «что сейчас». Он живёт своим кэшем: карта рождения
# не меняется никогда, а небо над ней — каждый день.
TRANSIT_V = 3

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
        "lilith_true": (ch.get("lilith_true") or {}).get("label", ""),
    }


# ── Темы разбора ─────────────────────────────────────────────────────
# Людям нужен не список фактов, а ответы: какой я, что мне подходит в
# работе, как я люблю. Под каждую тему здесь отбираются факторы, которые
# ремесло традиционно к ней относит. Отбор делает код, а не модель: так
# толкование профессий опирается на середину неба и шестой дом, а не на
# то, что модели показалось заметным.
PERSONAL = ("Солнце", "Луна", "Меркурий", "Венера", "Марс")

THEMES = [
    ("character", "Характер"),
    ("feelings", "Чувства и что даёт опору"),
    ("mind", "Ум и общение"),
    ("work", "Работа и призвание"),
    ("love", "Любовь и близость"),
    ("growth", "Сильные стороны и слабые места"),
]


def _cusp_sign(ch: dict, n: int) -> str:
    return ch["houses"][n - 1]["label"].split()[0]


def _house_line(ch: dict, n: int) -> list[str]:
    """Знак на куспиде, где стоит его управитель и кто сидит в самом доме."""
    by = {p["name"]: p for p in ch["planets"]}
    sign = _cusp_sign(ch, n)
    modern, classic = engine.RULER[sign]
    out = [f"{n}-й дом начинается в знаке {sign}"]
    for who in filter(None, (modern, classic)):
        p = by.get(who)
        if p:
            tag = "управитель" if who == modern else "управитель по старой традиции"
            out.append(f"{tag} {n}-го дома {who} стоит {p['label']}, в {p['house']}-м доме")
    inside = [p["name"] for p in ch["planets"] if p["house"] == n and not p.get("fictional")]
    out.append(f"в {n}-м доме: " + (", ".join(inside) if inside else "планет нет"))
    return out


def _aspects_of(ch: dict, names, limit: int = 5, skip_gen: bool = True) -> list[str]:
    names = set(names)
    out = []
    for a in ch["aspects"]:
        if a["a"] not in names and a["b"] not in names:
            continue
        if skip_gen and a["a"] in SLOW and a["b"] in SLOW:
            continue
        out.append(f"{a['a']} {a['type']} {a['b']} ({_arcmin(a['exact'])})")
        if len(out) >= limit:
            break
    return out


def _pos(ch: dict, name: str, houses: bool = True) -> str:
    p = next((x for x in ch["planets"] if x["name"] == name), None)
    if not p:
        return ""
    return (f"{name}: {p['label']}" + (f", {p['house']}-й дом" if houses else "")
            + (", ретроградна" if p["retro"] else ""))


def themes(ch: dict, time_known: bool = True) -> dict:
    d = digest(ch)
    # Время неизвестно — дома не называем вовсе: за сутки они проворачиваются
    # на полный круг, и «Солнце в седьмом доме» от полудня было бы выдумкой.
    _p = _pos
    _pos_ = lambda c, n: _p(c, n, time_known)
    el = ch.get("elements") or {}
    md = ch.get("modes") or {}
    t = {}

    t["character"] = [
        _pos_(ch, "Солнце"),
        *( [f"асцендент {ch['asc']['label']}", d["ruler"]] if time_known else [] ),
        _pos_(ch, "Луна"),
        "стихии (сколько тел): " + ", ".join(f"{k} {v}" for k, v in el.items()),
        "кресты: " + ", ".join(f"{k} {v}" for k, v in md.items()),
        *( ["пустые стихии: " + ", ".join(d["elements_missing"])] if d["elements_missing"] else [] ),
        *( ["скопления: " + "; ".join(d["stelliums"])] if d["stelliums"] else [] ),
        *_aspects_of(ch, ["Солнце"], 3),
    ]
    t["feelings"] = [
        _pos_(ch, "Луна"),
        d["moon_phase"],
        *_aspects_of(ch, ["Луна"], 4),
        *( _house_line(ch, 4) if time_known else [] ),
    ]
    t["mind"] = [
        _pos_(ch, "Меркурий"),
        *_aspects_of(ch, ["Меркурий"], 4),
        *( _house_line(ch, 3) if time_known else [] ),
    ]
    work = [_pos_(ch, "Солнце"), _pos_(ch, "Сатурн"), _pos_(ch, "Юпитер"), _pos_(ch, "Марс")]
    if time_known:
        work = [f"середина неба (MC) {ch['mc']['label']}", *_house_line(ch, 10),
                *_house_line(ch, 6), *_house_line(ch, 2), *work]
        work += [f"{a['a']} {a['type']} {a['b']} ({_arcmin(a['exact'])})"
                 for a in (ch.get("angle_aspects") or []) if a["b"] == "MC"][:3]
    t["work"] = work
    love = [_pos_(ch, "Венера"), _pos_(ch, "Марс"), *_aspects_of(ch, ["Венера", "Марс"], 5)]
    if time_known:
        love += [*_house_line(ch, 7), *_house_line(ch, 5)]
    t["love"] = love

    # Сильное и слабое ищем только среди аспектов, где участвует личная
    # планета: угол между двумя медленными был у всех, кто родился в тот год.
    personal = lambda a: a["a"] in PERSONAL or a["b"] in PERSONAL
    soft = [a for a in ch["aspects"] if a["type"] in ("тригон", "секстиль") and personal(a)][:4]
    hard = [a for a in ch["aspects"] if a["type"] in ("квадрат", "оппозиция") and personal(a)][:4]
    conj = [a for a in ch["aspects"] if a["type"] == "соединение"
            and (a["a"] in PERSONAL or a["b"] in PERSONAL)][:3]
    fmt = lambda a: f"{a['a']} {a['type']} {a['b']} ({_arcmin(a['exact'])})"
    t["growth"] = (["гармоничные аспекты: " + "; ".join(map(fmt, soft))] if soft else []) + \
                  (["напряжённые аспекты: " + "; ".join(map(fmt, hard))] if hard else []) + \
                  (["соединения с личными планетами: " + "; ".join(map(fmt, conj))] if conj else []) + \
                  (["ретроградны: " + ", ".join(d["retro"])] if d["retro"] else []) + \
                  (["на станции: " + ", ".join(d["stations"])] if d["stations"] else [])
    return {k: [x for x in v if x] for k, v in t.items()}


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

SYSTEM_PROFILE = (
    "Ты пишешь личный разбор натальной карты для страницы «Эфемерида». Человек "
    "пришёл узнать про себя: какой у него характер, что ему подходит в работе, "
    "как он любит. Отвечай на это прямо и по существу.\n"
    "\n"
    + VOICE +
    "Здесь голос чуть теплее обычного: человек читает про себя. Но без сюсюканья "
    "и без лести. Обращайся на «вы».\n"
    "\n"
    "Образец раздела. Карта чужая, факты оттуда брать нельзя — нужен только тон "
    "и устройство:\n"
    "\n"
    "«**Вам нужна работа, где результат можно потрогать.** Середина неба в Тельце, "
    "а её управитель Венера стоит в шестом доме — в доме ежедневного труда. Ремесло "
    "читает это как призвание к делу, у которого есть материал и ощутимый итог. "
    "По-житейски: отчёт о проделанной работе вас не утешит, а готовый стол, "
    "сведённый бюджет или вылеченный сад — утешит.\n"
    "\n"
    "Сатурн в квадрате к Солнцу добавляет медленный разгон. Первые годы в любом "
    "деле идут туго, зато то, что сложилось, держится десятилетиями. Бросать "
    "на втором году — ровно та ошибка, которую эта карта совершает чаще всего.»\n"
    "\n"
    "КАК ПИСАТЬ КАЖДЫЙ РАЗДЕЛ:\n"
    "— Первая фраза — вывод, прямой ответ на тему раздела, выделенный **жирным**. "
    "Не факт, а вывод: «Вам нужна работа…», «Вы влюбляетесь медленно…».\n"
    "— Потом — на чём вывод держится: какие именно положения карты, с числами "
    "из выданных фактов.\n"
    "— Потом — как это выглядит в жизни, обычными словами, с бытовым примером.\n"
    "— Два абзаца, всего 90–150 слов. Опирайся на два-три самых весомых фактора "
    "темы, а не перечисляй все.\n"
    "\n"
    "ПРАВИЛА, нарушать которые нельзя:\n"
    "1. Все знаки, дома, градусы и аспекты — ТОЛЬКО из выданных фактов. "
    "Ничего не пересчитывай и не добавляй.\n"
    "2. Толкуй уверенно, как принято в ремесле. Оговорка, что астрология — "
    "традиция, а не наука, уже стоит на странице: не повторяй её в тексте.\n"
    "3. Никаких предсказаний событий и дат: ни «выйдете замуж», ни «разбогатеете», "
    "ни «ждёт развод». Ты описываешь склонности, а не судьбу.\n"
    "4. Никаких диагнозов и советов про здоровье, лекарства, вложения денег.\n"
    "5. Никакой лести: не «вы уникальны», а что именно у вас получается и почему. "
    "Слабые места называй так же прямо, как сильные, и говори, что с ними делать.\n"
    "6. Ни один раздел не должен пересказывать другой. Солнце и асцендент — "
    "в «Характере», дальше к ним не возвращайся без нового угла.\n"
    "7. Разметка — только **жирный** для первой фразы раздела.\n"
)

PROFILE_PARTS = [
    # Две половины пишутся параллельно: вдвое быстрее, и каждая получает
    # модель, которая думает о трёх темах, а не о шести.
    {
        "ids": ("character", "feelings", "mind"),
        "extra": (
            "Кроме разделов верни \"summary\" — две фразы о человеке в целом, "
            "самое главное из всей карты, без чисел и без жирного."
        ),
        "shape": '{"sections": {"character": "текст", "feelings": "текст", "mind": "текст"}, '
                 '"summary": "две фразы"}',
    },
    {
        "ids": ("work", "love", "growth"),
        "extra": (
            "Для раздела work верни ещё \"fields\" — от пяти до семи конкретных "
            "сфер и профессий, которые традиция связывает с этой картой. Каждая — "
            "одно-три слова: «инженер-конструктор», «юриспруденция», «ресторанное "
            "дело». Не «творчество» и не «работа с людьми» — это ни о чём. "
            "Сферы должны следовать из фактов раздела work, а в тексте раздела "
            "объясни, из чего выбраны две-три главные."
        ),
        "shape": '{"sections": {"work": "текст", "love": "текст", "growth": "текст"}, '
                 '"fields": ["сфера", "сфера", "сфера", "сфера", "сфера"]}',
    },
]

TOPIC_HINT = {
    "character": "характер: как человек устроен, чем движим, как его видят другие",
    "feelings": "чувства: как переживает, что ему нужно для покоя, что выбивает из колеи",
    "mind": "ум и общение: как думает, учится, спорит, объясняет",
    "work": "работа и призвание: в какой работе раскроется, какая среда подходит, "
            "какая выматывает, как у него с деньгами как с привычкой (не как с прогнозом)",
    "love": "любовь и близость: как влюбляется, что ищет в партнёре, из-за чего "
            "ссорится, что ему нужно, чтобы остаться",
    "growth": "сильные стороны и слабые места: что даётся легко и на что опереться; "
              "где спотыкается и как это обходить",
}


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


def _profile_part(part: dict, t: dict, fict_warn: str, unknown_note: str, ask) -> dict:
    facts = {TOPIC_HINT[i]: t[i] for i in part["ids"]}
    prompt = ("Факты карты, уже разложенные по темам:\n" +
              json.dumps(facts, ensure_ascii=False, indent=1) + unknown_note + fict_warn +
              "\n\nНапиши по разделу на каждую тему. " + part["extra"] +
              "\n\nФормат ответа — строго JSON без пояснений:\n" + part["shape"])
    try:
        raw = ask(prompt, system=SYSTEM_PROFILE, temperature=0.85) or ""
    except Exception as e:
        print(f"⚠️ Разбор {part['ids']} не сгенерировался: {e}")
        return {}
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    try:
        out = json.loads(m.group(0))
    except Exception:
        return {}
    return out if isinstance(out, dict) else {}


def interpret(ch: dict, ask, time_known: bool = True) -> dict:
    d = digest(ch)
    lead = _lead(ch) if time_known else _lead_no_time(ch)
    t = themes(ch, time_known)
    fict = FICT_WARNING if any(p.get("fictional") for p in ch["planets"]) else ""
    unknown = ("" if time_known else
               "\n\nВАЖНО. Время рождения неизвестно, карта построена на полдень. "
               "Домов, асцендента и середины неба нет — не упоминай их. Луна может "
               "стоять на семь градусов в любую сторону: если она близко к границе "
               "знака, говори о ней осторожно.")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(PROFILE_PARTS)) as pool:
        outs = list(pool.map(lambda p: _profile_part(p, t, fict, unknown, ask), PROFILE_PARTS))

    titles = dict(THEMES)
    sections, summary = [], ""
    for part, out in zip(PROFILE_PARTS, outs):
        got = out.get("sections") if isinstance(out.get("sections"), dict) else {}
        for i in part["ids"]:
            text = got.get(i)
            if not isinstance(text, str) or len(text.strip()) < 40:
                continue
            sec = {"id": i, "title": titles[i], "text": text.strip()[:1500]}
            if i == "work":
                fields = [str(f).strip()[:40] for f in (out.get("fields") or [])
                          if isinstance(f, (str, int)) and str(f).strip()]
                if fields:
                    sec["fields"] = fields[:8]
            sections.append(sec)
        if isinstance(out.get("summary"), str):
            summary = out["summary"].strip()[:400]
    # держим порядок тем, даже если половины пришли вперемешку
    order = [i for i, _ in THEMES]
    sections.sort(key=lambda s: order.index(s["id"]))

    if not sections:
        return {**_fallback_reading(d), "lead": lead, "sections": [], "summary": ""}
    return {"lead": lead, "summary": summary, "sections": sections,
            "blocks": [], "verdict": ""}


def _lead_no_time(ch: dict) -> str:
    by = {p["name"]: p for p in ch["planets"]}
    return (f'Солнце {IN_SIGN.get(by["Солнце"]["sign"], "в " + by["Солнце"]["sign"])}, '
            f'Луна {IN_SIGN.get(by["Луна"]["sign"], "в " + by["Луна"]["sign"])}. '
            'Время рождения не указано, поэтому без асцендента и домов.')

def _trust_hits(tr: dict | None, time_known: bool) -> dict | None:
    """Без времени рождения углы карты взяты с полудня, а натальная Луна
    гуляет на семь градусов в любую сторону. Транзит «к вашему MC» или
    «к вашей Луне» в орбе два-три градуса тогда — чистая выдумка."""
    if not tr or time_known:
        return tr
    return {**tr, "hits": [h for h in tr["hits"] if h["to"] not in ("ASC", "MC", "Луна")]}


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


# Одна и та же оговорка нужна в двух местах: и в разборе карты, и в разделе
# «что сейчас». В авестийском режиме транзиты бьют и по Прозерпине с Селеной,
# так что предупреждать надо оба раза, иначе в одном тексте они честные
# конструкции, а в другом — молчаливые планеты.
FICT_WARNING = (
    "\n\nВАЖНО. Прозерпина и Селена — точки авестийской школы. Физических тел "
    "за ними нет, их никто не наблюдал: это конструкции с назначенными орбитами. "
    "Если упоминаешь их, скажи об этом прямо в том же абзаце, как говоришь про "
    "поколенческие аспекты. Ни одного слова, из которого следует, что это планеты."
)


def interpret_now(tr: dict, ask, fict: bool = False) -> dict:
    hits = " ".join(h["to"] for h in tr.get("hits", []))
    warn = FICT_WARNING if fict and ("Прозерпина" in hits or "Селена" in hits) else ""
    prompt = ("Факты:\n" + json.dumps(now_facts(tr), ensure_ascii=False, indent=1) +
              warn + "\n\nНапиши раздел «что сейчас» по правилам. Только JSON.")
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
    "карта рождения, разложенная по темам, и сегодняшнее небо над ней.\n"
    "\n" + VOICE + "\n"
    "ГЛАВНОЕ. Человек спросил — ответь. Первая же фраза — прямой ответ по карте, "
    "выделенный **жирным**. Не встречный вопрос, не рассуждение о том, что карта "
    "ничего не знает, — ответ, как его дал бы толковый астролог.\n"
    "\n"
    "КАК ОТВЕЧАТЬ НА РАЗНОЕ:\n"
    "— Про характер, таланты, работу, профессию, отношения, деньги как привычку — "
    "отвечай по существу и конкретно. Спросили, какая профессия подходит, — назови "
    "три-пять конкретных и объясни, из каких положений карты они следуют.\n"
    "— Про «что меня ждёт», «когда», «какой будет осень» — опирайся на выданные "
    "транзиты и их даты. Говори о периоде и его теме: «до 14 октября Сатурн давит "
    "на ваше Солнце — время, когда всё идёт с усилием». Конкретных событий не "
    "обещай: ни встреч, ни свадеб, ни увольнений.\n"
    "— Про «стоит ли», «что делать» — скажи, что в карте говорит за, что против, "
    "и в какой период это легче. Решение за человеком — скажи это одной короткой "
    "фразой, без нотаций.\n"
    "— Про здоровье, лекарства, суды, вложения — ответь, что по этой теме видно в "
    "карте, одной-двумя фразами, и прямо скажи, что решать такое надо с врачом, "
    "юристом или финансистом.\n"
    "\n"
    "ЗАПРЕЩЕНО:\n"
    "1. Придумывать знаки, дома, градусы и даты. Только выданные факты.\n"
    "2. Обещать события и исходы: «он вернётся», «вас повысят», «разбогатеете».\n"
    "3. Пугать, льстить и набивать цену. Никакой мистики, «кармы» и «вы особенный».\n"
    "4. Повторять оговорку, что астрология не наука, — она уже есть на странице.\n"
    "\n"
    "ЕСЛИ В ВОПРОСЕ БЕДА. Насилие, мысли о смерти, тяжёлая болезнь, отчаяние — "
    "брось астрологию совсем. Не считай, не толкуй, не шути. Скажи прямо и коротко, "
    "по-человечески: карта тут ничем не поможет, и с таким стоит идти к живому "
    "человеку — тому, кому доверяешь, или к специалисту. Три-четыре предложения, "
    "без глифов и без градусов.\n"
    "\n"
    "ОБЪЁМ. Два-четыре коротких абзаца. Разметка: **жирный** только для первой "
    "фразы. Никакого JSON, обычный текст, абзацы через пустую строку."
)

def answer_question(ch: dict, tr: dict | None, question: str, ask,
                    time_known: bool = True) -> str:
    t = themes(ch, time_known)
    facts = {
        "карта рождения по темам": {TOPIC_HINT[k]: v for k, v in t.items()},
        "самые точные аспекты": [
            f"{a['a']} {a['type']} {a['b']} ({_arcmin(a['exact'])})"
            for a in ch["aspects"][:6]],
    }
    if time_known:
        facts["углы карты"] = {"асцендент": ch["asc"]["label"], "середина неба": ch["mc"]["label"]}
    if tr:
        facts["сегодня"] = now_facts(tr)
    note = ("" if time_known else
            "\n\nВремя рождения неизвестно: домов, асцендента и середины неба нет, "
            "не упоминай их.")
    warn = FICT_WARNING if any(p.get("fictional") for p in ch["planets"]) else ""
    prompt = ("Факты:\n" + json.dumps(facts, ensure_ascii=False, indent=1) + note + warn +
              "\n\nВопрос человека: " + question.strip() +
              "\n\nОтветь по правилам.")
    try:
        out = (ask(prompt, system=SYSTEM_ASK, temperature=0.8) or "").strip()
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
    # Сбой модели приходит строкой «⚠️ …» — показывать её человеку как ответ нельзя.
    if out.startswith("⚠️"):
        print(f"⚠️ Модель вернула ошибку вместо ответа: {out[:200]}")
        return ""
    return out[:3000]

# ── Ручки ────────────────────────────────────────────────────────────
SCHOOLS = ("classic", "avestan")


class ChartIn(BaseModel):
    date: str = ""        # 1990-05-17
    time: str = "12:00"   # 10:50
    lat: float = 0.0
    lon: float = 0.0
    tz: str = ""
    place: str = ""       # как показать место в шапке
    unknown_time: bool = False
    school: str = "classic"


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
        if request is not None and not rate_ok(client_ip(request), 60, "geo"):
            return {"cities": [], "error": "Слишком часто. Подождите минуту."}
        return {"cities": geo_search(q)}

    @router.post("/chart")
    def chart_endpoint(body: ChartIn, request: Request):
        if engine is None:
            return {"error": "Расчёт временно недоступен: " + ENGINE_ERROR}
        if not rate_ok(client_ip(request), 20, "natal"):
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

        school = body.school if body.school in SCHOOLS else "classic"
        key = (f"v{READING_V}|{school}|{body.date}|{body.time}|{round(body.lat, 3)}"
               f"|{round(body.lon, 3)}|{tz}|{int(bool(body.unknown_time))}")
        # Предупреждение про перевод стрелок считается заново на каждый ответ:
        # оно выводится из тех же данных и стоит доли миллисекунды, зато старые
        # строки в кэше не надо пересчитывать ради нового поля.
        warn = tz_note(when, tz, body.lat, body.lon) if engine else ""

        cached = _cache_get("natal_readings", "key", key)

        try:
            ch = engine.chart(when, tz, body.lat, body.lon, school=school)
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
        tr = _trust_hits(tr, not body.unknown_time)

        now_key = f"t{TRANSIT_V}|{key}|{tr['when']}" if tr else ""
        now_cached = _cache_get("transit_readings", "key", now_key) if tr else None

        if cached is not None:
            reading = cached.get("reading") or {}
        if now_cached is not None and cached is not None:
            now_text = now_cached
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_read = (pool.submit(interpret, ch, ask, not body.unknown_time)
                          if cached is None else None)
                fict = any(p.get("fictional") for p in ch["planets"])
                f_now = (pool.submit(interpret_now, tr, ask, fict)
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
        if not rate_ok(client_ip(request), 8, "ask"):
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
            ch = engine.chart(when, tz, body.lat, body.lon,
                              school=body.school if body.school in SCHOOLS else "classic")
        except Exception as e:
            print(f"⚠️ Карта не посчиталась: {e}")
            return {"error": "Расчёт не сошёлся."}
        try:
            tr = engine.transits(ch)
        except Exception:
            tr = None
        tr = _trust_hits(tr, not body.unknown_time)

        text = answer_question(ch, tr, q, ask, not body.unknown_time)
        if not text:
            return {"error": "Ответ не сложился. Попробуйте ещё раз."}
        return {"question": q, "answer": text}

    return router
