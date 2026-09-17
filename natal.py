"""Натальная карта: честный расчёт.

Положения планет и куспиды домов считает Swiss Ephemeris в режиме Moshier —
он не требует файлов эфемерид, поэтому сервис остаётся лёгким. Точность
проверена сверкой с astropy: расхождение меньше минуты дуги, тогда как
аспекты меряются градусами.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import swisseph as swe

FLAGS = swe.FLG_SWIEPH | swe.FLG_MOSEPH | swe.FLG_SPEED

SIGNS = ["Овен", "Телец", "Близнецы", "Рак", "Лев", "Дева",
         "Весы", "Скорпион", "Стрелец", "Козерог", "Водолей", "Рыбы"]

PLANETS = [
    ("Солнце", swe.SUN), ("Луна", swe.MOON), ("Меркурий", swe.MERCURY),
    ("Венера", swe.VENUS), ("Марс", swe.MARS), ("Юпитер", swe.JUPITER),
    ("Сатурн", swe.SATURN), ("Уран", swe.URANUS), ("Нептун", swe.NEPTUNE),
    ("Плутон", swe.PLUTO), ("Сев. узел", swe.TRUE_NODE),
]

# Аспект — угол между планетами. Орб — допуск, в пределах которого
# астрология считает угол состоявшимся.
ASPECTS = [("соединение", 0, 8), ("секстиль", 60, 5), ("квадрат", 90, 7),
           ("тригон", 120, 7), ("оппозиция", 180, 8)]


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
            "house": _house_of(longitude, cusps),
        })

    return {
        "utc": utc.isoformat(),
        "jd": jd,
        "planets": bodies,
        "houses": [{"n": i + 1, "lon": round(c % 360, 4), "label": _fmt(c)}
                   for i, c in enumerate(cusps)],
        "asc": {"lon": round(ascmc[0] % 360, 4), "label": _fmt(ascmc[0])},
        "mc": {"lon": round(ascmc[1] % 360, 4), "label": _fmt(ascmc[1])},
        "aspects": _aspects(bodies),
    }


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
