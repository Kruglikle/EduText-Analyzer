from __future__ import annotations

from typing import Dict, List
import math
import re

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


def compute_textstat_metrics(text: str) -> Dict:
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
