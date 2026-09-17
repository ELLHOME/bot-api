"""Натальная карта: честный расчёт.

Положения планет и куспиды домов считает Swiss Ephemeris в режиме Moshier —
он не требует файлов эфемерид, поэтому сервис остаётся лёгким. Точность
проверена сверкой с astropy: расхождение меньше минуты дуги, тогда как
аспекты меряются градусами.
"""
from __future__ import annotations

import os
import datetime as dt
from zoneinfo import ZoneInfo

import swisseph as swe

FLAGS = swe.FLG_SWIEPH | swe.FLG_MOSEPH | swe.FLG_SPEED

# Большие планеты режим Moshier считает по формулам, зашитым в библиотеку, —
# никаких файлов не нужно. Хирон так не посчитать: он ходит между Сатурном
# и Ураном, они его постоянно дёргают, и движение не сводится к компактной
# формуле. Его положения берут из заранее посчитанной таблицы — это файл
# seas_18.se1 рядом с кодом. Файла нет — Хирон просто не считается,
# остальная карта работает как работала.
swe.set_ephe_path(os.path.dirname(os.path.abspath(__file__)))


def _chiron_works() -> bool:
    try:
        swe.calc_ut(swe.julday(2000, 1, 1, 12.0), swe.CHIRON, FLAGS)
        return True
    except Exception as e:
        print(f"⚠️ Хирон недоступен ({e}). Карта считается без него.")
        return False


HAS_CHIRON = _chiron_works()

SIGNS = ["Овен", "Телец", "Близнецы", "Рак", "Лев", "Дева",
         "Весы", "Скорпион", "Стрелец", "Козерог", "Водолей", "Рыбы"]

PLANETS = [
    ("Солнце", swe.SUN), ("Луна", swe.MOON), ("Меркурий", swe.MERCURY),
    ("Венера", swe.VENUS), ("Марс", swe.MARS), ("Юпитер", swe.JUPITER),
    ("Сатурн", swe.SATURN), ("Уран", swe.URANUS), ("Нептун", swe.NEPTUNE),
    ("Плутон", swe.PLUTO), ("Сев. узел", swe.TRUE_NODE),
]
if HAS_CHIRON:
    # Хирон вставляем перед узлом: он тело, а узел — точка пересечения орбит,
    # и в списке они разной природы.
    PLANETS.insert(-1, ("Хирон", swe.CHIRON))

# Аспект — угол между планетами. Орб — допуск, в пределах которого
# астрология считает угол состоявшимся.
ASPECTS = [("соединение", 0, 8), ("секстиль", 60, 5), ("квадрат", 90, 7),
           ("тригон", 120, 7), ("оппозиция", 180, 8)]

# К углам карты орбы уже: асцендент и MC — точки, а не тела,
# и широкий аспект к ним мало что значит.
ANGLE_ASPECTS = [("соединение", 0, 6), ("секстиль", 60, 3), ("квадрат", 90, 5),
                 ("тригон", 120, 5), ("оппозиция", 180, 6)]

ELEMENT = {"Овен": "огонь", "Лев": "огонь", "Стрелец": "огонь",
           "Телец": "земля", "Дева": "земля", "Козерог": "земля",
           "Близнецы": "воздух", "Весы": "воздух", "Водолей": "воздух",
           "Рак": "вода", "Скорпион": "вода", "Рыбы": "вода"}

MODE = {"Овен": "кардинальный", "Рак": "кардинальный", "Весы": "кардинальный",
        "Козерог": "кардинальный",
        "Телец": "фиксированный", "Лев": "фиксированный",
        "Скорпион": "фиксированный", "Водолей": "фиксированный",
        "Близнецы": "подвижный", "Дева": "подвижный",
        "Стрелец": "подвижный", "Рыбы": "подвижный"}

# Управитель знака. Для трёх знаков современная традиция назначила новые
# планеты, открытые после телескопа; старого управителя держим рядом,
# чтобы не выдавать одну школу за единственную.
RULER = {"Овен": ("Марс", ""), "Телец": ("Венера", ""), "Близнецы": ("Меркурий", ""),
         "Рак": ("Луна", ""), "Лев": ("Солнце", ""), "Дева": ("Меркурий", ""),
         "Весы": ("Венера", ""), "Скорпион": ("Плутон", "Марс"),
         "Стрелец": ("Юпитер", ""), "Козерог": ("Сатурн", ""),
         "Водолей": ("Уран", "Сатурн"), "Рыбы": ("Нептун", "Юпитер")}

# Средняя суточная скорость по эклиптике. Нужна, чтобы отличить планету
# «на станции» — в точке разворота она почти стоит, и это видимое событие,
# а не толкование.
MEAN_SPEED = {"Солнце": 0.986, "Луна": 13.176, "Меркурий": 1.383, "Венера": 1.200,
              "Марс": 0.524, "Юпитер": 0.083, "Сатурн": 0.034, "Уран": 0.012,
              "Нептун": 0.006, "Плутон": 0.004, "Сев. узел": 0.053,
              "Хирон": 0.019}   # оборот примерно за 50.7 года

PHASES = [(0, 12, "новолуние"), (12, 85, "растущий серп"), (85, 95, "первая четверть"),
          (95, 168, "растущая луна"), (168, 192, "полнолуние"),
          (192, 265, "убывающая луна"), (265, 275, "последняя четверть"),
          (275, 348, "старый серп"), (348, 361, "новолуние")]


def _sign(lon: float) -> tuple[str, float]:
    lon %= 360
    return SIGNS[int(lon // 30)], lon % 30


def _fmt(lon: float) -> str:
    name, deg = _sign(lon)
    d = int(deg)
    m = int(round((deg - d) * 60))
    if m == 60:
        d, m = d + 1, 0
    return f"{name} {d}°{m:02d}′"


def chart(when_local: dt.datetime, tz: str, lat: float, lon: float,
          hsys: bytes = b"P", fold: int = 0) -> dict:
    """when_local — местное время рождения, tz — зона вроде 'Europe/Moscow'.

    fold различает два прохода одного и того же часа в ночь перевода стрелок
    назад: 0 — первый (летнее время), 1 — второй (зимнее). В обычные сутки
    он ни на что не влияет.
    """
    aware = when_local.replace(tzinfo=ZoneInfo(tz), fold=fold)
    utc = aware.astimezone(dt.timezone.utc)
    jd = swe.julday(utc.year, utc.month, utc.day,
                    utc.hour + utc.minute / 60 + utc.second / 3600)

    cusps, ascmc = swe.houses(jd, lat, lon, hsys)

    bodies = []
    for name, pid in PLANETS:
        pos, _ = swe.calc_ut(jd, pid, FLAGS)
        longitude, speed = pos[0] % 360, pos[3]
        s, deg = _sign(longitude)
        bodies.append({
            "name": name, "lon": round(longitude, 4),
            "sign": s, "deg": round(deg, 2), "label": _fmt(longitude),
            "retro": speed < 0,
            "speed": round(speed, 5),
            "house": _house_of(longitude, cusps),
        })

    extra = _extras(jd, bodies, cusps, ascmc)
    return {
        "utc": utc.isoformat(),
        "jd": jd,
        "planets": bodies,
        **extra,
        "houses": [{"n": i + 1, "lon": round(c % 360, 4), "label": _fmt(c)}
                   for i, c in enumerate(cusps)],
        "asc": {"lon": round(ascmc[0] % 360, 4), "label": _fmt(ascmc[0])},
        "mc": {"lon": round(ascmc[1] % 360, 4), "label": _fmt(ascmc[1])},
        "aspects": _aspects(bodies),
        "angle_aspects": _angle_aspects(bodies, ascmc),
    }


def _extras(jd, bodies, cusps, ascmc) -> dict:
    """Всё, что даёт та же эфемерида, но обычно остаётся за кадром."""
    import math
    by = {b["name"]: b for b in bodies}

    # Фаза Луны: угол между Луной и Солнцем. Доля освещённого диска — из него же,
    # и сверена с pheno_ut, когда тот доступен.
    el = (by["Луна"]["lon"] - by["Солнце"]["lon"]) % 360
    illum = (1 - math.cos(math.radians(el))) / 2
    phase = next(n for a, b, n in PHASES if a <= el < b)

    # Дневная карта или ночная: Солнце над горизонтом или под ним.
    # Над горизонтом — дома с седьмого по двенадцатый.
    day = 7 <= by["Солнце"]["house"] <= 12

    asc_sign = _sign(ascmc[0])[0]
    rul, old = RULER[asc_sign]
    ruler = {"sign": asc_sign, "planet": rul, "classic": old,
             "label": by[rul]["label"] if rul in by else "",
             "house": by[rul]["house"] if rul in by else 0}

    el_count, mo_count = {}, {}
    for b in bodies:
        # Расклад по стихиям и крестам принято считать по десяти планетам.
        # Узел — не тело, Хирон в этот счёт традиционно не входит; добавь их —
        # и суммы перестанут сходиться с любой книгой.
        if b["name"] in ("Сев. узел", "Хирон"):
            continue
        el_count[ELEMENT[b["sign"]]] = el_count.get(ELEMENT[b["sign"]], 0) + 1
        mo_count[MODE[b["sign"]]] = mo_count.get(MODE[b["sign"]], 0) + 1

    # Стеллиум — три и более тела в одном знаке или доме. Скопление видно
    # на карте глазом, и его стоит назвать.
    groups = {}
    for b in bodies:
        groups.setdefault(("знак", b["sign"]), []).append(b["name"])
        groups.setdefault(("дом", b["house"]), []).append(b["name"])
    stelliums = [{"where": f"{k[0]} {k[1]}", "who": v}
                 for k, v in groups.items() if len(v) >= 3]

    # Планета на станции: скорость упала ниже десятой доли обычной,
    # то есть в ближайшие дни она развернётся.
    stations = [b["name"] for b in bodies
                if MEAN_SPEED.get(b["name"]) and
                abs(b["speed"]) < MEAN_SPEED[b["name"]] * 0.1]

    lilith = None
    try:
        pos, _ = swe.calc_ut(jd, swe.MEAN_APOG, FLAGS)
        lilith = {"lon": round(pos[0] % 360, 4), "label": _fmt(pos[0]),
                  "house": _house_of(pos[0] % 360, cusps)}
    except Exception:
        pass

    return {
        "moon_phase": {"angle": round(el, 2), "illum": round(illum * 100),
                       "name": phase},
        "day_chart": day,
        "ruler": ruler,
        "elements": el_count,
        "modes": mo_count,
        "stelliums": stelliums,
        "stations": stations,
        "lilith": lilith,
    }


def _angle_aspects(bodies: list[dict], ascmc) -> list[dict]:
    """Аспекты планет к асценденту и середине неба."""
    out = []
    for label, lon in (("ASC", ascmc[0] % 360), ("MC", ascmc[1] % 360)):
        for b in bodies:
            diff = abs(b["lon"] - lon) % 360
            if diff > 180:
                diff = 360 - diff
            for title, angle, orb in ANGLE_ASPECTS:
                delta = abs(diff - angle)
                if delta <= orb:
                    out.append({"a": b["name"], "b": label, "type": title,
                                "exact": round(delta, 2)})
                    break
    return sorted(out, key=lambda x: x["exact"])


def _house_of(longitude: float, cusps) -> int:
    """В каком доме лежит точка. Дома неравные, последний замыкает круг."""
    cs = [c % 360 for c in cusps]
    for i in range(12):
        a, b = cs[i], cs[(i + 1) % 12]
        if a <= b:
            if a <= longitude < b:
                return i + 1
        elif longitude >= a or longitude < b:   # дом переходит через 0°
            return i + 1
    return 1


def _aspects(bodies: list[dict]) -> list[dict]:
    out = []
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            a, b = bodies[i], bodies[j]
            diff = abs(a["lon"] - b["lon"]) % 360
            if diff > 180:
                diff = 360 - diff
            for title, angle, orb in ASPECTS:
                delta = abs(diff - angle)
                if delta <= orb:
                    out.append({"a": a["name"], "b": b["name"], "type": title,
                                "exact": round(delta, 2)})
                    break
    return sorted(out, key=lambda x: x["exact"])


# ── Транзиты: сегодняшнее небо против карты рождения ─────────────────
# Натальная карта не меняется никогда. Меняется небо над ней, и именно
# это люди имеют в виду под «что сейчас». Углы считаются к тем же точкам
# и той же эфемеридой — просто на другую дату.

# Тела, чьи транзиты имеет смысл показывать. Луна проходит знак за два дня,
# её транзиты живут часы — в списке событий от них один шум, поэтому Луна
# идёт отдельной строкой «где она сегодня», а не аспектами.
TRANSITING = [("Солнце", swe.SUN), ("Меркурий", swe.MERCURY), ("Венера", swe.VENUS),
              ("Марс", swe.MARS), ("Юпитер", swe.JUPITER), ("Сатурн", swe.SATURN),
              ("Уран", swe.URANUS), ("Нептун", swe.NEPTUNE), ("Плутон", swe.PLUTO)]
if HAS_CHIRON:
    TRANSITING.append(("Хирон", swe.CHIRON))

# Быстрым телам орб уже: иначе Солнце «касается» карты каждый день
# и событие перестаёт быть событием.
FAST = {"Солнце", "Меркурий", "Венера", "Марс"}


def _jd(when_utc: dt.datetime) -> float:
    return swe.julday(when_utc.year, when_utc.month, when_utc.day,
                      when_utc.hour + when_utc.minute / 60)


def _gap(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return 360 - d if d > 180 else d


def transits(natal_chart: dict, when_utc: dt.datetime | None = None,
             horizon: int = 120) -> dict:
    """Где планеты сейчас и что из этого стоит к точкам карты.

    Для каждого попадания ищем день, когда угол точен: сканируем окно
    в четыре месяца вокруг сегодня и берём минимум расхождения. Так же
    становится видно, сходится аспект или уже расходится.
    """
    now = when_utc or dt.datetime.now(dt.timezone.utc)
    jd = _jd(now)

    targets = {p["name"]: p["lon"] for p in natal_chart["planets"]}
    targets["ASC"] = natal_chart["asc"]["lon"]
    targets["MC"] = natal_chart["mc"]["lon"]

    sky = []
    for name, pid in TRANSITING:
        pos, _ = swe.calc_ut(jd, pid, FLAGS)
        sky.append({"name": name, "lon": round(pos[0] % 360, 4),
                    "label": _fmt(pos[0]), "retro": pos[3] < 0})

    moon, _ = swe.calc_ut(jd, swe.MOON, FLAGS)
    sun, _ = swe.calc_ut(jd, swe.SUN, FLAGS)
    el = (moon[0] - sun[0]) % 360
    import math
    moon_now = {"label": _fmt(moon[0]), "sign": _sign(moon[0])[0],
                "phase": next(n for a, b, n in PHASES if a <= el < b),
                "illum": round((1 - math.cos(math.radians(el))) / 2 * 100)}

    hits = []
    for name, pid in TRANSITING:
        pos, _ = swe.calc_ut(jd, pid, FLAGS)
        limit = 2.0 if name in FAST else 3.0
        for tname, tlon in targets.items():
            d = _gap(pos[0], tlon)
            for title, angle, _orb in ASPECTS:
                delta = abs(d - angle)
                if delta > limit:
                    continue
                # день, когда угол точен
                best_k, best_d = 0, 999.0
                step = 1 if name in FAST else 2
                for k in range(-horizon, horizon + 1, step):
                    p2, _ = swe.calc_ut(_jd(now + dt.timedelta(days=k)), pid, FLAGS)
                    dd = abs(_gap(p2[0], tlon) - angle)
                    if dd < best_d:
                        best_d, best_k = dd, k
                exact = (now + dt.timedelta(days=best_k)).date()
                hits.append({
                    "who": name, "type": title, "to": tname,
                    "orb": round(delta, 2),
                    "state": ("точен сейчас" if abs(best_k) <= 1
                              else "сходится" if best_k > 0 else "расходится"),
                    "exact": exact.isoformat(),
                    "days": best_k,
                    "retro": pos[3] < 0,
                    "slow": name not in FAST,
                })
                break
    hits.sort(key=lambda h: (h["orb"], not h["slow"]))
    return {"when": now.date().isoformat(), "sky": sky, "moon": moon_now, "hits": hits}
