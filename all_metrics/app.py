from __future__ import annotations

import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .config import load_settings, setup_logging
from .pipeline import run_analysis


settings = load_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="EduText Analyzer Backend", version="0.1.0")


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    enable_lev: bool = Form(True),
):
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")
    try:
        raw = await file.read()
        text = raw.decode("utf-8", errors="replace")
        # TODO: Move run_analysis to a background worker (Celery/RQ) for async processing.
        result = run_analysis(text, settings, enable_lev=enable_lev)
        return result
    except Exception as exc:
        logger.exception("Analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail="Analysis failed.")
