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
import urllib.error
import urllib.request
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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
FITNESS_RATE_MAX = int(os.getenv("FITNESS_RATE_MAX", "6"))  # анализов за 10 мин с одного IP

ANGLE_TITLES = {
    "front": "Фото спереди (фронт)",
    "side": "Фото сбоку (профиль)",
    "back": "Фото со спины",
    "side2": "Фото со второго бока",
}


class FitnessAnalyzeIn(BaseModel):
    profile: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    photos: dict[str, str] = {}
    catalog: list[dict[str, Any]] = []


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
        parts.append({"text": f"Ракурс: {ANGLE_TITLES.get(angle, angle)}"})
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


def build_router(rate_ok, client_ip) -> APIRouter:
    router = APIRouter(prefix="/fitness", tags=["fitness"])

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
        if not rate_ok(client_ip(request), FITNESS_RATE_MAX, "fitness"):
            return JSONResponse(
                {
                    "ok": False,
                    "error": "Слишком много AI-разборов подряд. Подождите несколько минут — локальный анализ работает без ограничений.",
                },
                status_code=429,
            )
        status, payload = analyze_body(body.profile, body.metrics, body.photos, body.catalog)
        return JSONResponse(payload, status_code=status)

    return router
