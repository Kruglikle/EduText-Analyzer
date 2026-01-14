from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple
from pathlib import Path
from functools import lru_cache
import logging
import time

import numpy as np
import pandas as pd
import Levenshtein
from transliterate import translit

from .config import Settings


logger = logging.getLogger(__name__)


def transliterate_ru_to_en(word_ru: str) -> str:
    try:
        return translit(str(word_ru), "ru", reversed=True).lower()
    except Exception:
        return ""


class ArgosTranslator:
    def __init__(
        self,
        model_path: Optional[Path],
        source_lang: str = "en",
        target_lang: str = "ru",
    ) -> None:
        self.model_path = model_path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._translator = None

    def ensure_ready(self) -> None:
        if self._translator is not None:
            return
        try:
            import argostranslate.package as argos_package
            import argostranslate.translate as argos_translate
        except Exception as exc:
            raise RuntimeError(
                "Argos Translate is not installed. Install argostranslate or switch "
                "EDUTEXT_TRANSLATOR_BACKEND to 'google'."
            ) from exc

        if self.model_path is not None and not self.model_path.exists():
            raise FileNotFoundError(f"Argos model not found: {self.model_path}")

        def find_translation():
            languages = argos_translate.get_installed_languages()
            src = next((lang for lang in languages if lang.code == self.source_lang), None)
            tgt = next((lang for lang in languages if lang.code == self.target_lang), None)
            if not src or not tgt:
                return None
            return src.get_translation(tgt)

        translator = find_translation()
        if translator is None and self.model_path is not None:
            argos_package.install_from_path(str(self.model_path))
            translator = find_translation()
        if translator is None:
            raise RuntimeError(
                f"Argos Translate model for {self.source_lang}->{self.target_lang} not installed. "
                "Provide EDUTEXT_ARGOS_MODEL_PATH to a local .argosmodel file or install the model."
            )
        self._translator = translator

    def translate(self, word: str) -> str:
        self.ensure_ready()
        try:
            return self._translator.translate(word) or ""
        except Exception as exc:
            logger.warning("Argos translation failed for '%s': %s", word, exc)
            return ""

    def translate_many(self, words: Iterable[str]) -> list[str]:
        self.ensure_ready()
        return [self.translate(w) for w in words]


@lru_cache(maxsize=2)
def _get_argos_translator_cached(model_path: Optional[str]) -> ArgosTranslator:
    path = Path(model_path) if model_path else None
    return ArgosTranslator(path)


def _get_google_translator(settings: Settings):
    try:
        from deep_translator import GoogleTranslator
    except Exception as exc:
        raise RuntimeError(
            "Google translator backend requested but deep-translator is not installed."
        ) from exc
    return GoogleTranslator(
        source="en", target="ru", timeout=settings.translate_timeout_sec
    )


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
    translator,
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
    for doc in nlp.pipe(
        texts,
        batch_size=settings.spacy_batch_size,
        n_process=settings.spacy_n_process,
    ):
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
    backend = settings.translator_backend
    translator = None
    if backend == "argos":
        translator = _get_argos_translator_cached(
            str(settings.argos_model_path) if settings.argos_model_path else None
        )
        translator.ensure_ready()
    elif backend == "google":
        translator = _get_google_translator(settings)
    else:
        raise ValueError(f"Unknown translator backend: {backend}")

    tr_map: Dict[str, str] = {}
    missing = [w for w in top_words if w not in cache]
    if missing:
        logger.info("Translating %d words (cache miss)", len(missing))
    translated_since_flush = 0
    flush_every = settings.translate_cache_flush_size
    for w in top_words:
        if w in cache:
            tr_map[w] = cache[w]
            continue
        if backend == "google":
            tr = translate_with_retry(
                translator, w, settings.translate_retries, settings.translate_sleep_sec
            )
            time.sleep(settings.translate_sleep_sec)
        else:
            tr = translator.translate(w)
        tr_map[w] = tr
        cache[w] = tr
        if flush_every and flush_every > 0:
            translated_since_flush += 1
            if translated_since_flush >= flush_every:
                save_translation_cache(settings.translate_cache_path, cache)
                translated_since_flush = 0

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
