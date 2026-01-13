# EduText-Analyzer
EduText Analyzer is a Python-based tool for automated analysis of school textbooks and educational texts in foreign languages. The project combines corpus linguistics, NLP, and machine learning to extract lexical, grammatical, and communicative features from texts.

## Запуск backend (FastAPI)

### 1) Установка зависимостей

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r all_metrics/requirements.txt
python -m spacy download en_core_web_sm
```

### 2) Запуск API

```bash
python -m uvicorn all_metrics.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3) Пример запроса

```bash
curl -F "file=@starlight-10.txt" -F "enable_lev=true" http://127.0.0.1:8000/analyze
```

Ответ содержит `job_id`, `status`, `paths` и `summary`. CSV сохраняются в:

```
all_metrics/runs/{job_id}/cefr_word_table.csv
all_metrics/runs/{job_id}/lev_words.csv
all_metrics/runs/{job_id}/exercises.csv
```

## Конфигурация (через переменные окружения)

- `EDUTEXT_CEFR_PATH` — путь к `len_cefr.csv` (по умолчанию корень репозитория).
- `EDUTEXT_MODEL_BUNDLE_EN` — путь к `model_bundle`.
- `EDUTEXT_MODEL_BUNDLE_RU` — путь к `model_bundle_ru`.
- `EDUTEXT_OUTPUT_DIR` — базовая папка для результатов (по умолчанию `all_metrics/`).
- `EDUTEXT_RUNS_DIR` — папка для `runs/`.
- `EDUTEXT_CACHE_DIR` — папка для кеша переводов.
- `EDUTEXT_TRANSLATE_LIMIT` — top‑N для перевода Levenshtein (по умолчанию 300).
- `EDUTEXT_LOG_LEVEL` — уровень логирования (например `INFO`).

## Тесты (минимум)

```bash
pytest all_metrics/tests -q
```
