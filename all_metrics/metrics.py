from __future__ import annotations

from typing import Dict, List, Tuple
from collections import OrderedDict
import hashlib
import math
import os
import re
import threading

import numpy as np
import pandas as pd
import textstat


def tokenize_en(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", (text or "").lower())


def compute_ttr_family(tokens: List[str], segment_len: int = 100) -> Dict:
    n = len(tokens)
    if n == 0:
        return {
            "tokens": 0,
            "types": 0,
            "ttr": np.nan,
            "rttr": np.nan,
            "cttr": np.nan,
            f"msttr_{segment_len}": np.nan,
        }
    types = len(set(tokens))
    ttr = types / n
    rttr = types / math.sqrt(n)
    cttr = types / math.sqrt(2 * n)
    n_full = n // segment_len
    msttr = (
        float(
            np.mean(
                [
                    len(set(tokens[i * segment_len : (i + 1) * segment_len]))
                    / segment_len
                    for i in range(n_full)
                ]
            )
        )
        if n_full > 0
        else np.nan
    )
    return {
        "tokens": n,
        "types": types,
        "ttr": ttr,
        "rttr": rttr,
        "cttr": cttr,
        f"msttr_{segment_len}": msttr,
    }


def safe_div(a, b):
    return a / b if b else np.nan


_TEXTSTAT_CACHE_MAXSIZE = int(os.environ.get("EDUTEXT_TEXTSTAT_CACHE_SIZE", 4096))
_TEXTSTAT_CACHE: "OrderedDict[Tuple[str, int], Dict]" = OrderedDict()
_TEXTSTAT_CACHE_LOCK = threading.Lock()


def _textstat_cache_key(text: str) -> Tuple[str, int]:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return digest, len(text)


def _compute_textstat_metrics_uncached(text: str) -> Dict:
    text = text or ""
    words = textstat.lexicon_count(text)
    sents = textstat.sentence_count(text)
    syll = textstat.syllable_count(text)
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text) if words else np.nan,
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text) if words else np.nan,
        "words_total": words,
        "sentences_total": sents,
        "syllables_total": syll,
        "avg_words_per_sentence": safe_div(words, sents),
    }


def _clear_textstat_cache() -> None:
    with _TEXTSTAT_CACHE_LOCK:
        _TEXTSTAT_CACHE.clear()


def compute_textstat_metrics(text: str) -> Dict:
    text = text or ""
    key = _textstat_cache_key(text)
    with _TEXTSTAT_CACHE_LOCK:
        cached = _TEXTSTAT_CACHE.get(key)
        if cached is not None:
            _TEXTSTAT_CACHE.move_to_end(key)
            return cached
    metrics = _compute_textstat_metrics_uncached(text)
    with _TEXTSTAT_CACHE_LOCK:
        _TEXTSTAT_CACHE[key] = metrics
        if len(_TEXTSTAT_CACHE) > _TEXTSTAT_CACHE_MAXSIZE:
            _TEXTSTAT_CACHE.popitem(last=False)
    return metrics


def compute_metrics_for_text(text: str, segment_len: int = 100) -> Dict:
    m = {}
    m.update(compute_textstat_metrics(text))
    m.update(compute_ttr_family(tokenize_en(text), segment_len=segment_len))
    return m


def compute_metrics_df_by_page(pages, segment_len=100, min_tokens=50, min_words=50) -> pd.DataFrame:
    rows = []
    for p in pages:
        m = compute_metrics_for_text(p.text_en, segment_len=segment_len)
        rows.append({"page_num": p.page_num, "module_id": p.module_id, **m})
    df = pd.DataFrame(rows).sort_values("page_num")
    df["is_sparse"] = (df["tokens"] < min_tokens) | (df["words_total"] < min_words)
    return df


def compute_metrics_df_by_module(by_page: pd.DataFrame) -> pd.DataFrame:
    return (
        by_page[~by_page["is_sparse"]]
        .groupby("module_id", dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
