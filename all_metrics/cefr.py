from __future__ import annotations

from typing import Dict, Tuple
from io import StringIO
import logging

import pandas as pd


logger = logging.getLogger(__name__)

CEFR_ORDER = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}


def load_cefr_lexicon(csv_path: str) -> Dict[str, str]:
    content = open(csv_path, "r", encoding="utf-8").read()
    if ";" in content and "," not in content:
        content = content.replace(";", ",")
    df = pd.read_csv(StringIO(content))

    word_col = None
    level_col = None
    for col in df.columns:
        col_norm = col.strip().lower()
        if col_norm == "word":
            word_col = col
        if col_norm in {"cefr level", "cefr_level"}:
            level_col = col
    if not word_col or not level_col:
        raise ValueError(
            f"CEFR CSV must have 'Word' and 'CEFR Level' columns, got: {df.columns.tolist()}"
        )

    word_levels = {}
    for _, row in df.iterrows():
        w = str(row[word_col]).strip().lower()
        lvl = str(row[level_col]).strip().upper()
        if not w:
            continue
        if w in word_levels:
            cur = word_levels[w]
            if CEFR_ORDER.get(lvl, 99) < CEFR_ORDER.get(cur, 99):
                word_levels[w] = lvl
        else:
            word_levels[w] = lvl
    logger.info("Loaded CEFR lexicon: %d words", len(word_levels))
    return word_levels


def compute_cefr_word_table(pages, word_levels: Dict[str, str], nlp) -> Tuple[pd.DataFrame, Dict]:
    level_names = ["A1", "A2", "B1", "B2", "C1", "C2"]
    lemma_freq = {}
    lemma_pos_counts = {}
    total_by_level = dict.fromkeys(level_names, 0)
    total_tokens = 0
    texts = [p.text_en or "" for p in pages]
    for doc in nlp.pipe(texts, batch_size=16):
        for tok in doc:
            if not tok.is_alpha or tok.is_stop:
                continue
            lemma = tok.lemma_.lower()
            total_tokens += 1
            lemma_freq[lemma] = lemma_freq.get(lemma, 0) + 1
            lemma_pos_counts.setdefault(lemma, {})
            lemma_pos_counts[lemma][tok.pos_] = lemma_pos_counts[lemma].get(tok.pos_, 0) + 1
            lvl = word_levels.get(lemma)
            if lvl in total_by_level:
                total_by_level[lvl] += 1
    rows = []
    for lemma, freq in lemma_freq.items():
        lvl = word_levels.get(lemma)
        if lvl not in CEFR_ORDER:
            continue
        pos_counts = lemma_pos_counts.get(lemma, {})
        pos = max(pos_counts.items(), key=lambda x: x[1])[0] if pos_counts else "X"
        rows.append({"word": lemma, "level": lvl, "frequency": freq, "pos": pos})
    cefr_word_table = pd.DataFrame(rows)
    if not cefr_word_table.empty:
        cefr_word_table["level_order"] = cefr_word_table["level"].map(CEFR_ORDER)
        cefr_word_table = (
            cefr_word_table.sort_values(["level_order", "frequency"], ascending=[True, False]).drop(
                columns=["level_order"]
            )
        )
    cefr_summary = {
        "tokens_total": int(total_tokens),
        "by_level_tokens": {lvl: int(total_by_level[lvl]) for lvl in level_names},
        "by_level_pct": {
            lvl: (total_by_level[lvl] / total_tokens) if total_tokens else None for lvl in level_names
        },
    }
    return cefr_word_table, cefr_summary
