from __future__ import annotations

from typing import Dict, Optional, Tuple
from pathlib import Path
import logging
import time

import numpy as np
import pandas as pd
import Levenshtein
from deep_translator import GoogleTranslator
from transliterate import translit

from .config import Settings


logger = logging.getLogger(__name__)


def transliterate_ru_to_en(word_ru: str) -> str:
    try:
        return translit(str(word_ru), "ru", reversed=True).lower()
    except Exception:
        return ""


def load_translation_cache(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to read translation cache %s: %s", path, exc)
        return {}
    cache = {}
    for _, row in df.iterrows():
        w = str(row.get("word", "")).strip().lower()
        tr = str(row.get("translation_ru", "")).strip()
        if w:
            cache[w] = tr
    return cache


def save_translation_cache(path: Path, cache: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"word": w, "translation_ru": tr} for w, tr in sorted(cache.items())]
    pd.DataFrame(rows).to_csv(path, index=False)


def translate_with_retry(
    translator: GoogleTranslator,
    word: str,
    retries: int,
    sleep_sec: float,
) -> str:
    for attempt in range(retries + 1):
        try:
            return translator.translate(word) or ""
        except Exception as exc:
            if attempt >= retries:
                logger.warning("Translation failed for '%s': %s", word, exc)
                return ""
            time.sleep(sleep_sec * (attempt + 1))
    return ""


def compute_lev_words_all(
    pages,
    nlp,
    settings: Settings,
    translate_limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict]:
    freq: Dict[str, int] = {}
    texts = [p.text_en or "" for p in pages]
    for doc in nlp.pipe(texts, batch_size=16):
        for tok in doc:
            if not tok.is_alpha or tok.is_stop:
                continue
            lemma = tok.lemma_.lower()
            if len(lemma) == 1 and lemma != "i":
                continue
            freq[lemma] = freq.get(lemma, 0) + 1
    all_words = sorted(freq.keys())

    if translate_limit is None:
        translate_limit = settings.translate_limit
    limit = int(translate_limit)
    top_words = (
        [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:limit]]
        if limit > 0
        else []
    )

    cache = load_translation_cache(settings.translate_cache_path)
    translator = GoogleTranslator(
        source="en", target="ru", timeout=settings.translate_timeout_sec
    )

    tr_map: Dict[str, str] = {}
    missing = [w for w in top_words if w not in cache]
    if missing:
        logger.info("Translating %d words (cache miss)", len(missing))
    for w in top_words:
        if w in cache:
            tr_map[w] = cache[w]
            continue
        tr = translate_with_retry(
            translator, w, settings.translate_retries, settings.translate_sleep_sec
        )
        tr_map[w] = tr
        cache[w] = tr
        time.sleep(settings.translate_sleep_sec)

    if missing:
        save_translation_cache(settings.translate_cache_path, cache)

    rows = []
    for w in all_words:
        tr_ru = tr_map.get(w, "")
        tr_en = transliterate_ru_to_en(tr_ru) if tr_ru else ""
        sim = Levenshtein.ratio(w, tr_en) if tr_en else np.nan
        rows.append(
            {
                "word": w,
                "frequency": int(freq[w]),
                "translation_ru": tr_ru,
                "translation_en_translit": tr_en,
                "similarity": sim,
            }
        )
    lev_words = pd.DataFrame(rows).sort_values("frequency", ascending=False).reset_index(drop=True)
    lev_summary = {
        "words_total": int(len(all_words)),
        "translated": int(len(top_words)),
        "lev_mean_over_translated": float(np.nanmean(lev_words["similarity"]))
        if len(top_words)
        else None,
    }
    return lev_words, lev_summary
