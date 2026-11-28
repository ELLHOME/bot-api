"""
Фитнес-анализ фигуры и осанки по фото (BodyCoach AI) для сайта ELLHOME.

Принимает до 4 сжатых фото (спереди, сбоку, со спины, второй бок),
локальные CV-замеры пропорций и каталог упражнений; вызывает Gemini Vision
через серверный GOOGLE_API_KEY (ключ никогда не попадает в браузер и в git)
и возвращает структурированный разбор фигуры и персональный многодневный план.
Фото не сохраняются на диск и не логируются.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

# Порядок моделей: сначала явно заданная переменная, затем основная модель сервера,
# затем актуальные запасные модели Gemini Vision.
VISION_MODELS: list[str] = []
for _m in (
    os.getenv("GEMINI_FITNESS_MODEL"),
    os.getenv("GEMINI_MODEL"),
    "gemini-3.6-flash",
    "gemini-4-hydrogen",
    "gemini-3.5-flash-lite",
):
    if _m and _m.strip() and _m.strip() not in VISION_MODELS:
        VISION_MODELS.append(_m.strip())

ALLOWED_ANGLES = ("front", "side", "back", "side2")
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}
MAX_PHOTOS = 4
MAX_PHOTO_BYTES = 1_600_000          # ~1.6 МБ на одно сжатое фото (клиент шлёт ~60-150 КБ)
FITNESS_RATE_MAX = int(os.getenv("FITNESS_RATE_MAX", "6"))  # анализов фигуры за 10 мин с одного IP
MEAL_RATE_MAX = int(os.getenv("MEAL_RATE_MAX", "15"))       # анализов еды за 10 мин с одного IP
# Окно в 10 минут не мешает растянуть перебор на весь день, поэтому есть и суточные
# лимиты на IP, и общий потолок AI-запросов за сутки (UTC) — он ограничивает счёт за Gemini.
FITNESS_DAILY_MAX = int(os.getenv("FITNESS_DAILY_MAX", "20"))  # анализов фигуры в сутки с одного IP
MEAL_DAILY_MAX = int(os.getenv("MEAL_DAILY_MAX", "60"))        # анализов еды в сутки с одного IP
AI_DAILY_CAP = int(os.getenv("FITNESS_AI_DAILY_CAP", "400"))   # всех AI-запросов фитнеса в сутки

_daily: dict[str, int] = {}
_daily_day = ""


def _daily_take(ip: str, bucket: str, ip_limit: int) -> str:
    """Проверяет и списывает суточные лимиты. Возвращает "" или причину отказа: "ip" / "cap"."""
    global _daily_day
    day = time.strftime("%Y-%m-%d", time.gmtime())
    if day != _daily_day:
        _daily.clear()
        _daily_day = day
    ip_key = f"{bucket}|{ip}"
    if _daily.get("*", 0) >= AI_DAILY_CAP:
        return "cap"
    if _daily.get(ip_key, 0) >= ip_limit:
        return "ip"
    _daily["*"] = _daily.get("*", 0) + 1
    _daily[ip_key] = _daily.get(ip_key, 0) + 1
    return ""

ANGLE_TITLES = {
    "front": "Фото спереди (фронт)",
    "side": "Фото сбоку (профиль)",
    "back": "Фото со спины",
    "side2": "Фото со второго бока",
}

MEAL_TYPE_LABELS = {
    "breakfast": "Завтрак",
    "lunch": "Обед",
    "dinner": "Ужин",
    "snack": "Перекус",
}


class FitnessAnalyzeIn(BaseModel):
    profile: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    photos: dict[str, str] = {}
    catalog: list[dict[str, Any]] = []


class MealAnalyzeIn(BaseModel):
    photo: str = ""
    note: str = ""
    mealType: str = "lunch"
    profile: dict[str, Any] = {}
    dailyTarget: dict[str, Any] = {}


def _parse_data_url(data_url: str) -> tuple[str, str]:
    match = re.match(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", str(data_url or ""), re.DOTALL)
    if not match:
        raise ValueError("Некорректный формат изображения (ожидается data:image/...;base64,...)")
    mime = match.group(1).lower()
    if mime == "image/jpg":
        mime = "image/jpeg"
    if mime not in ALLOWED_MIMES:
        raise ValueError(f"Неподдерживаемый формат фото: {mime}")
    b64_data = re.sub(r"\s+", "", match.group(2))
    # Оцениваем декодированный размер без лишних копий
    est_bytes = (len(b64_data) * 3) // 4
    if est_bytes < 500:
        raise ValueError("Изображение слишком маленькое или пустое.")
    if est_bytes > MAX_PHOTO_BYTES:
        raise ValueError("Одно из фото слишком большое (лимит 1.5 МБ после сжатия).")
    # Быстрая проверка валидности base64
    base64.b64decode(b64_data[:128] + "====", validate=False)
    return mime, b64_data


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = json.loads(cleaned[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Модель вернула ответ не в формате JSON.")


def _build_vision_prompt(
    profile: dict[str, Any],
    metrics: dict[str, Any],
    catalog: list[dict[str, Any]],
) -> str:
    days_count = max(3, min(5, int(profile.get("daysPerWeek") or 3)))
    safe_profile = {
        "sex": str(profile.get("sex") or "male")[:20],
        "age": profile.get("age"),
        "heightCm": profile.get("heightCm"),
        "weightKg": profile.get("weightKg"),
        "goal": str(profile.get("goal") or "recomp")[:30],
        "equipment": str(profile.get("equipment") or "home_bodyweight")[:30],
        "level": str(profile.get("level") or "beginner")[:20],
        "daysPerWeek": days_count,
    }
    safe_metrics = {
        "shoulderToWaist": metrics.get("shoulderToWaist"),
        "waistToHip": metrics.get("waistToHip"),
        "shoulderTiltDeg": metrics.get("shoulderTiltDeg"),
        "hipTiltDeg": metrics.get("hipTiltDeg"),
        "forwardHeadDeg": metrics.get("forwardHeadDeg"),
        "thoracicRoundScore": metrics.get("thoracicRoundScore"),
        "pelvicTiltHint": metrics.get("pelvicTiltHint"),
        "postureScore": metrics.get("postureScore"),
        "anglesProvided": metrics.get("anglesProvided", []),
    }
    catalog_lines = "\n".join(
        f'- {str(c.get("id", ""))[:40]}: {str(c.get("name", ""))[:80]} '
        f'[{str(c.get("category", ""))[:20]}, инвентарь: {",".join(str(x) for x in (c.get("equipment") or [])[:4])}]'
        for c in (catalog or [])[:50]
        if isinstance(c, dict) and c.get("id")
    )
    return (
        "Ты — элитный персональный фитнес-тренер и специалист по биомеханике и осанке. "
        "Тебе переданы фотографии пользователя с разных ракурсов (спереди, сбоку, со спины) "
        "и локальные CV-замеры пропорций и углов осанки.\n\n"
        "Задача:\n"
        "1) Проведи внимательный, уважительный и конкретный разбор фигуры по фото: "
        "пропорции плеч/талии/бёдер, визуальный мышечный тонус, симметрия плеч и таза, "
        "положение шеи, грудного отдела и поясницы. Без медицинских диагнозов.\n"
        f"2) Составь персональный план на {days_count} тренировочных дня строго из доступного каталога "
        f"упражнений ниже, подходящих под выбранный инвентарь ({safe_profile['equipment']}). "
        "В каждом дне 5–7 упражнений: 1–2 на осанку/разминку, 3–4 основных силовых, 1–2 на кор/финиш. "
        "ОБЯЗАТЕЛЬНО: все тренировочные дни должны быть РАЗНЫМИ по специализации и составу упражнений "
        "(никаких одинаковых дней вроде 2-го и 4-го), а в недельную программу обязательно включи "
        "отдельные упражнения на бицепс и на трицепс.\n\n"
        f"Параметры пользователя: {json.dumps(safe_profile, ensure_ascii=False)}\n"
        f"Локальные CV-замеры: {json.dumps(safe_metrics, ensure_ascii=False)}\n\n"
        f"Каталог упражнений приложения (используй ТОЛЬКО эти id):\n{catalog_lines}\n\n"
        "Верни СТРОГО валидный JSON на русском языке без markdown-обёртки со следующей структурой:\n"
        "{\n"
        '  "summary": "3–4 предложения развёрнутого разбора фигуры и осанки по фото",\n'
        '  "somatotype": "Характеристика телосложения и баланса пропорций",\n'
        '  "visualComposition": "Краткая тактичная оценка тонуса и композиции тела",\n'
        '  "strengths": ["Сильная сторона 1", "Сильная сторона 2", "Сильная сторона 3"],\n'
        '  "focusAreas": [\n'
        '    {"area": "Зона внимания", "observation": "Что заметно на фото", "priority": "high"}\n'
        "  ],\n"
        '  "postureNotes": ["Наблюдение по осанке сбоку/спереди 1", "Наблюдение 2"],\n'
        '  "trainingAdvice": ["Рекомендация по тренировкам 1", "Рекомендация 2", "Рекомендация 3"],\n'
        '  "nutritionHint": "Персональный ориентир по питанию, белку и восстановлению",\n'
        '  "recommendedExerciseIds": ["id1", "id2", "id3", "id4"],\n'
        '  "customPlanDays": [\n'
        "    {\n"
        '      "dayIndex": 1,\n'
        '      "title": "День 1 • Название фокуса дня",\n'
        '      "focus": "Короткое описание цели дня",\n'
        '      "items": [\n'
        "        {\n"
        '          "exerciseId": "wall_angels",\n'
        '          "section": "Разминка и осанка",\n'
        '          "sets": 3,\n'
        '          "reps": "12 повт.",\n'
        '          "restSec": 45,\n'
        '          "reason": "Конкретно почему выбрано по фото"\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}"
    )


def _call_gemini_vision(
    model: str,
    api_key: str,
    prompt: str,
    parsed_photos: list[tuple[str, str, str]],
    opener=None,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"text": prompt}]
    for angle, mime, b64_data in parsed_photos:
        label = "Фото блюда" if angle == "meal" else f"Ракурс: {ANGLE_TITLES.get(angle, angle)}"
        parts.append({"text": label})
        parts.append({"inlineData": {"mimeType": mime, "data": b64_data}})

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        # Для Gemini 3.5+/4 не задаём temperature и не зажимаем maxOutputTokens,
        # чтобы внутренние thought-токены не обрывали JSON-ответ.
        "generationConfig": {
            "responseMimeType": "application/json",
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with (opener or urllib.request.urlopen)(req, timeout=75) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            body = exc.read().decode("utf-8", errors="ignore")
            detail = json.loads(body).get("error", {}).get("message", "") or body[:220]
        except Exception:
            detail = ""
        raise RuntimeError(f"HTTP {exc.code}: {detail[:220]}") from None

    candidates = data.get("candidates") or []
    if not candidates:
        block_reason = (data.get("promptFeedback") or {}).get("blockReason")
        if block_reason:
            raise RuntimeError(f"Запрос отклонён фильтром безопасности модели ({block_reason})")
        raise RuntimeError("Модель вернула пустой ответ.")

    out_parts = candidates[0].get("content", {}).get("parts") or []
    text_chunks = [p.get("text", "") for p in out_parts if p.get("text") and not p.get("thought")]
    if not text_chunks:
        text_chunks = [p.get("text", "") for p in out_parts if p.get("text")]
    return _extract_json_object("\n".join(text_chunks))


def analyze_body(
    profile: dict[str, Any],
    metrics: dict[str, Any],
    photos: dict[str, str],
    catalog: list[dict[str, Any]],
    opener=None,
) -> tuple[int, dict[str, Any]]:
    key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key or "..." in key:
        return 503, {"ok": False, "error": "Серверный ключ AI Vision (GOOGLE_API_KEY) не настроен."}

    parsed_photos: list[tuple[str, str, str]] = []
    for angle in ALLOWED_ANGLES:
        raw_url = (photos or {}).get(angle)
        if not raw_url:
            continue
        try:
            mime, b64_data = _parse_data_url(raw_url)
        except ValueError as err:
            return 400, {"ok": False, "error": str(err)}
        parsed_photos.append((angle, mime, b64_data))
        if len(parsed_photos) >= MAX_PHOTOS:
            break

    if not parsed_photos:
        return 400, {"ok": False, "error": "Добавьте хотя бы одно фото для AI-анализа."}

    prompt = _build_vision_prompt(profile or {}, metrics or {}, catalog or [])
    errors: list[str] = []
    for model in VISION_MODELS:
        try:
            analysis = _call_gemini_vision(model, key, prompt, parsed_photos, opener=opener)
            return 200, {
                "ok": True,
                "provider": "gemini",
                "model": model,
                "analysis": analysis,
            }
        except Exception as exc:
            print(f"⚠️ Fitness Vision ({model}) не сработал: {exc}")
            errors.append(f"{model}: {exc}")
            continue

    return 502, {
        "ok": False,
        "error": "AI Vision сейчас временно недоступен. Локальный разбор и план уже готовы.",
        "detail": " · ".join(errors)[:360],
    }


def _build_meal_prompt(
    note: str,
    meal_type: str,
    profile: dict[str, Any],
    daily_target: dict[str, Any],
    has_photo: bool,
) -> str:
    safe_profile = {
        "goal": str((profile or {}).get("goal") or "recomp")[:30],
        "weightKg": (profile or {}).get("weightKg"),
        "sex": str((profile or {}).get("sex") or "male")[:20],
    }
    safe_target = {
        "targetKcal": (daily_target or {}).get("targetKcal"),
        "proteinGrams": (daily_target or {}).get("proteinGrams"),
        "fatGrams": (daily_target or {}).get("fatGrams"),
        "carbsGrams": (daily_target or {}).get("carbsGrams"),
    }
    clean_note = (note or "").strip()[:600]
    meal_label = MEAL_TYPE_LABELS.get(str(meal_type or "").lower(), "Приём пищи")
    source_desc = (
        "Тебе передано фото блюда" + (f" и пояснение пользователя: «{clean_note}»." if clean_note else ".")
        if has_photo
        else f"Пользователь описал блюдо текстом (без фото): «{clean_note}»."
    )
    return (
        "Ты — опытный спортивный нутрициолог. "
        f"{source_desc}\n"
        f"Тип приёма пищи: {meal_label}.\n"
        f"Профиль пользователя: {json.dumps(safe_profile, ensure_ascii=False)}\n"
        f"Дневные ориентиры: {json.dumps(safe_target, ensure_ascii=False)}\n\n"
        "Задача:\n"
        "1) Определи блюдо и разбей его на конкретные видимые/указанные ингредиенты (включая соусы, масло при жарке, хлеб, напитки, если видны или указаны).\n"
        "2) Для каждого ингредиента оцени реалистичный вес в граммах, калорийность (ккал), белки (г), жиры (г), углеводы (г) и клетчатку (г).\n"
        "3) Учитывай реальную калорийность ГОТОВЫХ блюд (например, варёная гречка/рис ~110–130 ккал на 100 г, а не сухая крупа 350 ккал/100 г).\n"
        "4) Дай короткий практичный комментарий под цель пользователя и 1–2 совета.\n\n"
        "Верни СТРОГО валидный JSON на русском языке без markdown-обёртки:\n"
        "{\n"
        '  "dishName": "Короткое понятное название блюда",\n'
        '  "confidence": "high" | "medium" | "low",\n'
        '  "portionNote": "Как оценена порция (например: стандартная тарелка ~24 см, ~380 г)",\n'
        '  "ingredients": [\n'
        "    {\n"
        '      "name": "Куриная грудка на гриле",\n'
        '      "grams": 160,\n'
        '      "kcal": 264,\n'
        '      "protein": 49.6,\n'
        '      "fat": 5.8,\n'
        '      "carbs": 0,\n'
        '      "fiber": 0\n'
        "    }\n"
        "  ],\n"
        '  "healthScore": 85,\n'
        '  "goalFeedback": "1–2 предложения оценки блюда под цель пользователя",\n'
        '  "tips": ["Короткий совет 1", "Короткий совет 2"]\n'
        "}"
    )


def _normalize_meal_result(raw: dict[str, Any]) -> dict[str, Any]:
    def num(val: Any, default: float = 0.0, lo: float = 0.0, hi: float = 5000.0) -> float:
        try:
            v = float(val)
            if v != v:  # NaN
                return default
            return max(lo, min(hi, v))
        except Exception:
            return default

    raw_ings = raw.get("ingredients") if isinstance(raw.get("ingredients"), list) else []
    ingredients: list[dict[str, Any]] = []
    for item in raw_ings[:20]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()[:80]
        if not name:
            continue
        grams = int(round(num(item.get("grams"), 100.0, 1.0, 2500.0)))
        kcal = int(round(num(item.get("kcal"), 0.0, 0.0, 4000.0)))
        protein = round(num(item.get("protein"), 0.0, 0.0, 300.0), 1)
        fat = round(num(item.get("fat"), 0.0, 0.0, 300.0), 1)
        carbs = round(num(item.get("carbs"), 0.0, 0.0, 500.0), 1)
        fiber = round(num(item.get("fiber"), 0.0, 0.0, 100.0), 1)
        if kcal == 0 and (protein > 0 or fat > 0 or carbs > 0):
            kcal = int(round(protein * 4 + fat * 9 + carbs * 4))
        ingredients.append({
            "name": name,
            "grams": grams,
            "kcal": kcal,
            "protein": protein,
            "fat": fat,
            "carbs": carbs,
            "fiber": fiber,
        })

    if not ingredients:
        t = raw.get("totals") if isinstance(raw.get("totals"), dict) else {}
        ingredients.append({
            "name": str(raw.get("dishName") or "Блюдо").strip()[:80] or "Блюдо",
            "grams": int(round(num(t.get("grams"), 250.0, 10.0, 2500.0))),
            "kcal": int(round(num(t.get("kcal"), 350.0, 0.0, 4000.0))),
            "protein": round(num(t.get("protein"), 20.0, 0.0, 300.0), 1),
            "fat": round(num(t.get("fat"), 12.0, 0.0, 300.0), 1),
            "carbs": round(num(t.get("carbs"), 35.0, 0.0, 500.0), 1),
            "fiber": round(num(t.get("fiber"), 4.0, 0.0, 100.0), 1),
        })

    totals = {
        "grams": int(sum(x["grams"] for x in ingredients)),
        "kcal": int(sum(x["kcal"] for x in ingredients)),
        "protein": round(sum(x["protein"] for x in ingredients), 1),
        "fat": round(sum(x["fat"] for x in ingredients), 1),
        "carbs": round(sum(x["carbs"] for x in ingredients), 1),
        "fiber": round(sum(x["fiber"] for x in ingredients), 1),
    }
    conf = str(raw.get("confidence") or "medium").strip().lower()
    if conf not in ("high", "medium", "low"):
        conf = "medium"
    tips = [
        str(x).strip()[:180]
        for x in (raw.get("tips") if isinstance(raw.get("tips"), list) else [])
        if str(x).strip()
    ][:3]
    return {
        "dishName": str(raw.get("dishName") or ingredients[0]["name"]).strip()[:100],
        "confidence": conf,
        "portionNote": str(raw.get("portionNote") or "").strip()[:180],
        "ingredients": ingredients,
        "totals": totals,
        "healthScore": int(round(num(raw.get("healthScore"), 78.0, 10.0, 100.0))),
        "goalFeedback": str(raw.get("goalFeedback") or "").strip()[:320],
        "tips": tips,
    }


def analyze_meal(
    photo: str,
    note: str,
    meal_type: str,
    profile: dict[str, Any],
    daily_target: dict[str, Any],
    opener=None,
) -> tuple[int, dict[str, Any]]:
    key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key or "..." in key:
        return 503, {"ok": False, "error": "Серверный ключ AI Vision (GOOGLE_API_KEY) не настроен."}

    clean_note = (note or "").strip()[:600]
    parsed_photos: list[tuple[str, str, str]] = []
    if photo and str(photo).strip():
        try:
            mime, b64_data = _parse_data_url(photo)
        except ValueError as err:
            return 400, {"ok": False, "error": str(err)}
        parsed_photos.append(("meal", mime, b64_data))

    if not parsed_photos and len(clean_note) < 3:
        return 400, {"ok": False, "error": "Загрузите фото блюда или напишите, что вы съели."}

    prompt = _build_meal_prompt(clean_note, meal_type, profile or {}, daily_target or {}, bool(parsed_photos))
    errors: list[str] = []
    for model in VISION_MODELS:
        try:
            raw = _call_gemini_vision(model, key, prompt, parsed_photos, opener=opener)
            meal = _normalize_meal_result(raw)
            return 200, {
                "ok": True,
                "provider": "gemini",
                "model": model,
                "meal": meal,
            }
        except Exception as exc:
            print(f"⚠️ Fitness Meal Vision ({model}) не сработал: {exc}")
            errors.append(f"{model}: {exc}")
            continue

    return 502, {
        "ok": False,
        "error": "AI-анализ еды сейчас временно недоступен. Попробуйте ещё раз через минуту.",
        "detail": " · ".join(errors)[:360],
    }


def build_router(rate_ok, client_ip, notify=None, record=None) -> APIRouter:
    router = APIRouter(prefix="/fitness", tags=["fitness"])

    def _limited(request: Request, bucket: str, burst_max: int, daily_max: int, burst_msg: str):
        ip = client_ip(request)
        if not rate_ok(ip, burst_max, bucket):
            return JSONResponse({"ok": False, "error": burst_msg}, status_code=429)
        reason = _daily_take(ip, bucket, daily_max)
        if reason == "ip":
            return JSONResponse(
                {"ok": False, "error": "Дневной лимит AI-анализов исчерпан. Попробуйте завтра — локальный анализ работает без ограничений."},
                status_code=429,
            )
        if reason == "cap":
            if notify:
                notify(
                    f"⚠️ Силуэтика: исчерпан общий суточный лимит AI-запросов ({AI_DAILY_CAP}). "
                    "До полуночи UTC AI-анализ отключён. Если это не наплыв людей, а перебор — проверьте логи Render.",
                    throttle_key=f"fitness_cap_{_daily_day}",
                    throttle_sec=86400,
                )
            return JSONResponse(
                {"ok": False, "error": "AI-анализ на сегодня временно недоступен из-за высокой нагрузки. Попробуйте завтра — локальный анализ работает без ограничений."},
                status_code=429,
            )
        return None

    def _recorded(request: Request, name: str, status: int, props: dict[str, Any]):
        """Запись в статистику уходит после ответа, чтобы база не задерживала анализ."""
        if not record:
            return None
        visitor = re.sub(r"[^A-Za-z0-9-]", "", request.headers.get("x-visitor-id") or "")[:40]
        return BackgroundTask(record, name, {"ok": status == 200, "status": status, **props}, visitor)

    @router.get("/health")
    def fitness_health(request: Request):
        if not rate_ok(client_ip(request), 60, "fhealth"):
            return JSONResponse({"status": "limited", "ready": False}, status_code=429)
        key = (os.getenv("GOOGLE_API_KEY") or "").strip()
        ready = bool(key and "..." not in key)
        return {
            "status": "ok",
            "ready": ready,
            "provider": "gemini",
            "model": VISION_MODELS[0] if VISION_MODELS else "",
        }

    @router.post("/analyze")
    def fitness_analyze(body: FitnessAnalyzeIn, request: Request):
        blocked = _limited(
            request, "fitness", FITNESS_RATE_MAX, FITNESS_DAILY_MAX,
            "Слишком много AI-разборов подряд. Подождите несколько минут — локальный анализ работает без ограничений.",
        )
        if blocked:
            return blocked
        status, payload = analyze_body(body.profile, body.metrics, body.photos, body.catalog)
        return JSONResponse(payload, status_code=status, background=_recorded(
            request, "fit_ai_body", status, {"photos": len(body.photos or {})}))

    @router.post("/meal")
    def fitness_meal(body: MealAnalyzeIn, request: Request):
        blocked = _limited(
            request, "fmeal", MEAL_RATE_MAX, MEAL_DAILY_MAX,
            "Слишком много запросов анализа еды подряд. Подождите несколько минут.",
        )
        if blocked:
            return blocked
        status, payload = analyze_meal(
            body.photo,
            body.note,
            body.mealType,
            body.profile,
            body.dailyTarget,
        )
        return JSONResponse(payload, status_code=status, background=_recorded(
            request, "fit_ai_meal", status, {"photo": bool(body.photo)}))

    return router
