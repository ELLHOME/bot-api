# ELLHOME bot API

Бэкенд AI-консультанта для сайта ELLHOME. Тот же «мозг», что и в учебном боте
(RAG + Router + Tool Calling + Memory + опц. Judge), но отдаёт JSON по HTTP.
Ключ модели живёт только на сервере и в код сайта не попадает.

## Контракт (что вызывает сайт)

`POST /chat`
```json
{ "message": "сколько стоит лендинг?", "history": [ {"role":"user","content":"привет"}, {"role":"assistant","content":"Здравствуйте!"} ] }
```
Ответ:
```json
{ "reply": "…", "category": "цена", "lead": null }
```
`GET /` — проверка живости: `{ "status": "ok", "provider": "gemini" }`.

## Локальный запуск (бесплатно, Ollama)

```bash
pip install -r requirements.txt
# нужно запущенное приложение Ollama и модель: ollama pull muse-glimmer (или llama3.2)
PROVIDER=ollama uvicorn server:app --reload
```
API поднимется на http://localhost:8000 (проверь http://localhost:8000/ ).

## Деплой на Render (облако)

1. Залей эту папку в отдельный репозиторий на GitHub.
2. На render.com → **New → Blueprint** → выбери этот репозиторий (подхватит `render.yaml`).
3. В настройках сервиса добавь переменную окружения **`GOOGLE_API_KEY`** — твой ключ Gemini
   (тот же, что в `.env` воркбука). `PROVIDER` уже задан = `gemini`.
4. (Опционально) Подключи Postgres и добавь **`DATABASE_URL`**, чтобы заявки переживали
   перезапуск сервера.
5. После деплоя получишь адрес вида `https://ellhome-bot-api.onrender.com` — его и укажет
   виджет на сайте.

⚠️ **Бесплатный тариф Render засыпает** после ~15 мин простоя: первое сообщение после сна
приходит с задержкой ~30–60 сек (сервер «просыпается»). Дальше — быстро. Для портфолио
это ок; если захочешь без задержки — есть платный always-on или пинг-раз-в-10-минут.

## Настройка под себя

- Факты, услуги и цены — в `knowledge.txt`.
- Разрешённые домены сайта (CORS) — список `ALLOWED_ORIGINS` в `server.py`
  (или переменная окружения `ALLOWED_ORIGINS`, домены через запятую).
- Проверка ответа «судьёй» — переменная окружения `USE_JUDGE=true`.
- Заявки: локально пишутся в `leads.json`; с `DATABASE_URL` — в Postgres.
