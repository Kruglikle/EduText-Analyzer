from __future__ import annotations

from typing import Dict, Optional
from pathlib import Path
import logging
import uuid

import pandas as pd
import spacy

from .config import Settings
from .preprocess import preprocess_document
from .metrics import compute_metrics_df_by_module, compute_metrics_df_by_page
from .cefr import load_cefr_lexicon, compute_cefr_word_table
from .lev import compute_lev_words_all
from .exercise_types import (
    extract_candidate_instructions,
    load_models,
    predict_ex_types_multilang,
)


logger = logging.getLogger(__name__)

_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _ensure_dirs(settings: Settings) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)


def run_analysis(
    text: str,
    settings: Settings,
    enable_lev: bool = True,
    job_id: Optional[str] = None,
) -> Dict:
    _ensure_dirs(settings)
    if not job_id:
        job_id = str(uuid.uuid4())
    run_dir = settings.runs_dir / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    nlp = get_nlp()
    pages = preprocess_document(text)
    logger.info("Preprocess complete: %d pages", len(pages))

    by_page = compute_metrics_df_by_page(pages)
    by_module = compute_metrics_df_by_module(by_page)

    word_levels = load_cefr_lexicon(str(settings.cefr_csv_path))
    cefr_word_table, cefr_summary = compute_cefr_word_table(pages, word_levels, nlp)

    lev_words = pd.DataFrame(
        columns=["word", "frequency", "translation_ru", "translation_en_translit", "similarity"]
    )
    lev_summary = None
    lev_status = "disabled"
    if enable_lev:
        lev_words, lev_summary = compute_lev_words_all(pages, nlp, settings)
        lev_status = "ok"

    exercises_df = pd.DataFrame(
        columns=["page_num", "module_id", "instruction", "lang", "chosen", "pred_label", "pred_conf"]
    )
    ex_status = "ok"
    try:
        en_bundle, ru_bundle = load_models(settings)
        ex_df = extract_candidate_instructions(pages)
        if len(ex_df) > 0:
            preds = predict_ex_types_multilang(
                ex_df["instruction"].astype(str).tolist(), ru_bundle, en_bundle
            )
            ex_df["lang"] = preds["lang"]
            ex_df["chosen"] = preds["chosen"]
            ex_df["pred_label"] = preds["final_pred"]
            ex_df["pred_conf"] = preds["final_conf"]
        exercises_df = ex_df[
            ["page_num", "module_id", "instruction", "lang", "chosen", "pred_label", "pred_conf"]
        ].copy()
    except Exception as exc:
        ex_status = "error"
        logger.error("Exercise types failed: %s", exc)

    output_cefr = run_dir / "cefr_word_table.csv"
    output_lev = run_dir / "lev_words.csv"
    output_ex = run_dir / "exercises.csv"

    cefr_word_table.to_csv(output_cefr, index=False)
    lev_words.to_csv(output_lev, index=False)
    exercises_df.to_csv(output_ex, index=False)

    status = "ok"
    if ex_status != "ok":
        status = "partial"

    summary = {
        "pages": len(pages),
        "modules": int(by_page["module_id"].nunique(dropna=False)),
        "cefr": cefr_summary,
        "lev": lev_summary,
        "lev_status": lev_status,
        "exercises": {"rows": int(len(exercises_df)), "status": ex_status},
    }

    return {
        "job_id": job_id,
        "status": status,
        "paths": {
            "cefr_word_table": str(output_cefr),
            "lev_words": str(output_lev),
            "exercises": str(output_ex),
        },
        "summary": summary,
        "run_dir": str(run_dir),
    }
