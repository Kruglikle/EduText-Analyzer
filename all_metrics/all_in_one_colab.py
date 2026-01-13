# ALL-IN-ONE: textbook.txt -> PREPROCESS -> METRICS + CEFR + LEV + EX TYPES (PRINT ONLY)
# LOCAL RUN, SAVE ONLY 3 CSV IN all_metrics/
# =========================
import os
import re
import math
import time
import pickle
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import textstat
import spacy
import Levenshtein
from deep_translator import GoogleTranslator
from transliterate import translit
from IPython.display import display

# ---------- PATHS ----------
def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "len_cefr.csv").exists():
            return p
    return start


REPO_ROOT = find_repo_root(Path.cwd())
OUTPUT_DIR = REPO_ROOT / "all_metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def resolve_textbook_path(repo_root: Path, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Textbook file not found: {p}")
        print("Textbook path (explicit):", p)
        return p

    candidates = []
    for search_dir in [repo_root, repo_root / "input_textbooks"]:
        if search_dir.exists():
            candidates.extend(sorted(search_dir.glob("*.txt")))

    if len(candidates) == 1:
        print("Textbook path (auto):", candidates[0])
        return candidates[0].resolve()

    if len(candidates) == 0:
        raise FileNotFoundError(
            "No textbook .txt files found. Set TEXTBOOK_PATH env var or place a .txt in repo root or input_textbooks."
        )

    raise FileNotFoundError(
        "Multiple .txt files found. Set TEXTBOOK_PATH env var to select one: "
        + ", ".join(str(p) for p in candidates)
    )


INPUT_TXT_PATH = os.environ.get("TEXTBOOK_PATH")
TXT_PATH = resolve_textbook_path(REPO_ROOT, INPUT_TXT_PATH)
CEFR_CSV_PATH = (REPO_ROOT / "len_cefr.csv").resolve()
MODEL_BUNDLE_EN_DIR = (REPO_ROOT / "model_bundle").resolve()
MODEL_BUNDLE_RU_DIR = (REPO_ROOT / "model_bundle_ru").resolve()

if not TXT_PATH.exists():
    raise FileNotFoundError(f"Missing {TXT_PATH}.")
if not CEFR_CSV_PATH.exists():
    raise FileNotFoundError(f"Missing {CEFR_CSV_PATH}. Place len_cefr.csv in {REPO_ROOT}")

print("Repo root:", REPO_ROOT)
print("TXT_PATH:", TXT_PATH)
print("CEFR_CSV_PATH:", CEFR_CSV_PATH)
print("OUTPUT_DIR:", OUTPUT_DIR)

ENABLE_CEFR = True
ENABLE_LEV = True

# Levenshtein: production top-k for translation
LEV_TRANSLATE_TOP_N = 2500

try:
    nlp = spacy.load("en_core_web_sm")
except OSError as e:
    raise RuntimeError(
        "SpaCy model en_core_web_sm is missing. Install with: python -m spacy download en_core_web_sm"
    ) from e

# =========================
# PREPROCESS (MVP)
# =========================
PAGE_RE_DEFAULT = r"^[=\-]{2,}\s*PAGE\s*(\d+)\s*[=\-]{2,}$"


@dataclass
class PageBlock:
    page_num: int
    text_raw: str
    text_clean: str
    text_en: str
    module_id: Optional[str] = None


def normalize_whitespace(s: str) -> str:
    s = (s or "").replace("\ufeff", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\t", " ")
    s = re.sub(r"[ \u00A0]+", " ", s)
    s = re.sub(r"[ \u00A0]+\n", "\n", s)
    return s.strip()


def dehyphenate_linebreaks(s: str) -> str:
    return re.sub(r"(?<=\w)-\n(?=\w)", "", s)


def looks_like_new_block(line: str) -> bool:
    t = (line or "").strip()
    if not t:
        return True
    if re.match(r"^\d+[\.\)]\s+", t):
        return True
    if re.match(r"^\d+\s*[a-zA-Z]\)\s+", t):
        return True
    if re.match(r"^\d+[a-zA-Z]\b", t):
        return True
    if re.match(r"^[\-\*]\s+", t):
        return True
    if re.match(r"^(Module|Unit|Lesson|Starter module|Модуль|Юнит|Урок)\b", t, re.IGNORECASE):
        return True
    letters = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", t)
    if letters and sum(ch.isupper() for ch in letters) / max(len(letters), 1) > 0.8 and len(letters) >= 6:
        return True
    return False


def join_wrapped_lines_preserve_paragraphs(s: str) -> str:
    lines = s.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            out.append("")
            i += 1
            continue
        merged = line.rstrip()
        j = i
        while j + 1 < len(lines):
            nxt = lines[j + 1]
            if nxt.strip() == "":
                break
            if looks_like_new_block(nxt):
                break
            if re.search(r"[.!?:;]$|[.!?:;][\"')\]]*$", merged.strip()):
                break
            merged = merged + " " + nxt.strip()
            j += 1
        out.append(merged)
        i = j + 1
    joined = "\n".join(out)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def extract_english_layer(s: str) -> str:
    tokens = re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[.!?,;:\-()\[\]\"']+|\n+", s
    )
    kept = []
    for tok in tokens:
        if tok.startswith("\n"):
            kept.append(tok)
            continue
        if re.search(r"[A-Za-z]", tok):
            kept.append(tok)
        elif re.match(r"^[0-9]+$", tok):
            kept.append(tok)
        elif re.match(r"^[.!?,;:\-()\[\]\"']+$", tok):
            kept.append(tok)
    out = " ".join(kept)
    out = out.replace(" \n ", "\n").replace(" \n", "\n").replace("\n ", "\n")
    out = re.sub(r"[ ]{2,}", " ", out)
    out = re.sub(r"\s+([.!?,;:])", r"\1", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def detect_module_id(text_clean: str) -> Optional[str]:
    for line in (text_clean or "").splitlines():
        t = line.strip()
        m = re.match(r"^(?:Module|Модуль)\s+([0-9IVXLC]+)$", t, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def split_into_pages(text: str, page_re: str = PAGE_RE_DEFAULT) -> List[Dict]:
    text = normalize_whitespace(text)
    page_re_comp = re.compile(page_re, re.MULTILINE)
    matches = list(page_re_comp.finditer(text))
    if not matches:
        return [{"page_num": 1, "text": text}]
    pages = []
    prefix = text[: matches[0].start()].strip()
    if prefix:
        pages.append({"page_num": 1, "text": prefix})
    for idx, m in enumerate(matches):
        page_num = int(m.group(1))
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        page_text = text[start:end].strip()
        pages.append({"page_num": page_num, "text": page_text})
    return pages


def preprocess_document(text_raw: str, page_re: str = PAGE_RE_DEFAULT):
    pages = split_into_pages(text_raw, page_re=page_re)
    out_pages = []
    last_module: Optional[str] = None
    for p in pages:
        raw = p["text"]
        s = normalize_whitespace(raw)
        s = dehyphenate_linebreaks(s)
        s = join_wrapped_lines_preserve_paragraphs(s)
        module_here = detect_module_id(s)
        if module_here:
            last_module = module_here
        en = extract_english_layer(s)
        out_pages.append(
            PageBlock(
                page_num=p["page_num"],
                text_raw=raw,
                text_clean=s,
                text_en=en,
                module_id=last_module,
            )
        )
    return out_pages


# =========================
# METRICS
# =========================
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


# =========================
# CEFR
# =========================
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
        raise ValueError(f"CEFR CSV must have 'Word' and 'CEFR Level' columns, got: {df.columns.tolist()}")

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


# =========================
# LEVENSHTEIN (words all, no cache, print only)
# - таблица полная по словам (кроме 1-буквенных, кроме i)
# - перевод делаем только для top-N, чтобы не умереть по лимитам
# =========================
def transliterate_ru_to_en(word_ru: str) -> str:
    try:
        return translit(str(word_ru), "ru", reversed=True).lower()
    except Exception:
        return ""


def compute_lev_words_all(pages, nlp, translate_limit: Optional[int] = None) -> Tuple[pd.DataFrame, Dict]:
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

    to_translate = []
    if translate_limit is None:
        to_translate = all_words
    else:
        limit = int(translate_limit)
        if limit > 0:
            top_words = [
                w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:limit]
            ]
            to_translate = top_words

    translator = GoogleTranslator(source="en", target="ru")
    tr_map: Dict[str, str] = {}
    for i, w in enumerate(to_translate, start=1):
        try:
            tr_map[w] = translator.translate(w)
        except Exception:
            tr_map[w] = ""
        if i % 50 == 0:
            time.sleep(0.2)
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
        "translated": int(len(to_translate)),
        "lev_mean_over_translated": float(np.nanmean(lev_words["similarity"])) if len(to_translate) else None,
    }
    return lev_words, lev_summary


# =========================
# EX TYPES  RU+EN ensemble with gating (PRINT ONLY)
# =========================
def _find_dir_by_name(root: str, name: str, max_depth: int = 4) -> Optional[str]:
    if not os.path.exists(root):
        return None
    for dirpath, dirnames, _ in os.walk(root):
        depth = os.path.relpath(dirpath, root).count(os.sep)
        if depth > max_depth:
            dirnames[:] = []
            continue
        if os.path.basename(dirpath) == name:
            return dirpath
    return None


def resolve_model_bundle_dirs(
    en_dir: str, ru_dir: str, search_root: Path
) -> Tuple[Optional[str], Optional[str]]:
    en_path = en_dir if os.path.exists(en_dir) else None
    ru_path = ru_dir if os.path.exists(ru_dir) else None

    if en_path and ru_path:
        print("Model bundles found in provided paths.")
        return en_path, ru_path

    print("Searching for model bundles in:", search_root)
    if not en_path:
        en_path = _find_dir_by_name(str(search_root), "model_bundle")
    if not ru_path:
        ru_path = _find_dir_by_name(str(search_root), "model_bundle_ru")
    if en_path and ru_path:
        print("Model bundles found via local search.")
    return en_path, ru_path


def _pick_file(bundle_dir: str, candidates: List[str], patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        for name in candidates:
            if re.fullmatch(pat, name, flags=re.IGNORECASE):
                return os.path.join(bundle_dir, name)
    for pat in patterns:
        for name in candidates:
            if re.search(pat, name, flags=re.IGNORECASE):
                return os.path.join(bundle_dir, name)
    if candidates:
        return os.path.join(bundle_dir, sorted(candidates)[0])
    return None


def load_sklearn_bundle(bundle_dir: str, label: str):
    if not os.path.isdir(bundle_dir):
        raise FileNotFoundError(f"{label}: bundle dir not found: {bundle_dir}")

    files = sorted(os.listdir(bundle_dir))
    print(f"[{label}] bundle dir: {bundle_dir}")
    print(f"[{label}] files: {files}")

    pkl_files = [f for f in files if f.lower().endswith(".pkl")]
    pipeline_candidates = [f for f in pkl_files if re.search(r"pipeline", f, re.IGNORECASE)]
    vectorizer_candidates = [f for f in pkl_files if re.search(r"(vectorizer|tfidf)", f, re.IGNORECASE)]
    model_candidates = [
        f for f in pkl_files if re.search(r"(model|clf|logreg)", f, re.IGNORECASE)
    ]

    pipeline_path = None
    if pipeline_candidates:
        pipeline_path = _pick_file(
            bundle_dir,
            pipeline_candidates,
            [r"^pipeline\.pkl$", r"pipeline"],
        )
    elif not vectorizer_candidates:
        pipeline_path = _pick_file(
            bundle_dir,
            [f for f in pkl_files if f.lower() in {"pipeline.pkl", "model.pkl", "clf.pkl"}],
            [r"^pipeline\.pkl$", r"^model\.pkl$", r"^clf\.pkl$"],
        )

    if pipeline_path:
        print(f"[{label}] loading mode: pipeline -> {os.path.basename(pipeline_path)}")
        try:
            with open(pipeline_path, "rb") as f:
                pipe = pickle.load(f)
            return None, pipe, True
        except Exception as e:
            print(f"[{label}] pipeline load failed: {repr(e)}")
            print(f"[{label}] dir contents: {files}")
            raise

    if not vectorizer_candidates or not model_candidates:
        raise FileNotFoundError(
            f"{label}: missing vectorizer/model in {bundle_dir}. Files: {files}"
        )

    vect_path = _pick_file(
        bundle_dir,
        vectorizer_candidates,
        [
            r"^tfidf_vectorizer\.pkl$",
            r"^vectorizer\.pkl$",
            r"tfidf",
            r"vectorizer",
        ],
    )
    model_path = _pick_file(
        bundle_dir,
        model_candidates,
        [r"^logreg_model\.pkl$", r"^model\.pkl$", r"^clf\.pkl$", r"logreg", r"model", r"clf"],
    )

    if not vect_path or not model_path:
        raise FileNotFoundError(
            f"{label}: could not resolve vectorizer/model in {bundle_dir}. Files: {files}"
        )

    print(
        f"[{label}] loading mode: vectorizer+clf -> {os.path.basename(vect_path)} + {os.path.basename(model_path)}"
    )
    try:
        with open(vect_path, "rb") as f:
            vect = pickle.load(f)
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        return vect, clf, False
    except Exception as e:
        print(f"[{label}] vectorizer/model load failed: {repr(e)}")
        print(f"[{label}] dir contents: {files}")
        raise


def detect_lang_simple(text: str) -> str:
    s = text or ""
    cyr = len(re.findall(r"[А-Яа-яЁё]", s))
    lat = len(re.findall(r"[A-Za-z]", s))
    if cyr == 0 and lat == 0:
        return "unknown"
    if cyr > lat:
        return "ru"
    if lat > cyr:
        return "en"
    return "mixed"


def predict_bundle(texts: List[str], vect, clf, is_pipeline: bool):
    if is_pipeline:
        pred = clf.predict(texts)
        conf = (
            clf.predict_proba(texts).max(axis=1)
            if hasattr(clf, "predict_proba")
            else np.full(len(texts), np.nan)
        )
        return pred, conf
    X = vect.transform(texts)
    pred = clf.predict(X)
    conf = (
        clf.predict_proba(X).max(axis=1)
        if hasattr(clf, "predict_proba")
        else np.full(len(texts), np.nan)
    )
    return pred, conf


def predict_ex_types_multilang(texts: List[str], ru_bundle, en_bundle):
    ru_vect, ru_clf, ru_is_pipe = ru_bundle
    en_vect, en_clf, en_is_pipe = en_bundle
    ru_pred, ru_conf = predict_bundle(texts, ru_vect, ru_clf, ru_is_pipe)
    en_pred, en_conf = predict_bundle(texts, en_vect, en_clf, en_is_pipe)
    final_pred, final_conf, chosen, lang_tag = [], [], [], []
    for i, t in enumerate(texts):
        lang = detect_lang_simple(t)
        lang_tag.append(lang)
        if lang == "ru":
            final_pred.append(ru_pred[i])
            final_conf.append(float(ru_conf[i]) if not np.isnan(ru_conf[i]) else np.nan)
            chosen.append("ru_lang")
        elif lang == "en":
            final_pred.append(en_pred[i])
            final_conf.append(float(en_conf[i]) if not np.isnan(en_conf[i]) else np.nan)
            chosen.append("en_lang")
        else:
            rc = float(ru_conf[i]) if not np.isnan(ru_conf[i]) else -1.0
            ec = float(en_conf[i]) if not np.isnan(en_conf[i]) else -1.0
            if rc >= ec:
                final_pred.append(ru_pred[i])
                final_conf.append(rc)
                chosen.append("ru_conf")
            else:
                final_pred.append(en_pred[i])
                final_conf.append(ec)
                chosen.append("en_conf")
    return {
        "ru_pred": ru_pred,
        "ru_conf": ru_conf,
        "en_pred": en_pred,
        "en_conf": en_conf,
        "final_pred": np.array(final_pred),
        "final_conf": np.array(final_conf),
        "chosen": np.array(chosen),
        "lang": np.array(lang_tag),
    }


def extract_candidate_instructions(pages) -> pd.DataFrame:
    """MVP: берём строки страниц, которые похожи на инструкции."""
    rows = []
    for p in pages:
        lines = [ln.strip() for ln in (p.text_clean or "").splitlines()]
        for ln in lines:
            if not ln or len(ln) < 5:
                continue
            if (
                re.match(r"^\d+[\)\.]?\s*", ln)
                or re.search(
                    r"\b(Match|Complete|Choose|Read|Listen|Fill|Write|Discuss|Answer|Look|Put|Find|Make)\b",
                    ln,
                    re.IGNORECASE,
                )
                or re.search(
                    r"\b(Соотнеси|Заполни|Выбери|Прочитай|Прослушай|Напиши|Ответь|Скажи|Составь|Найди|Поставь)\b",
                    ln,
                    re.IGNORECASE,
                )
            ):
                rows.append({"page_num": p.page_num, "module_id": p.module_id, "instruction": ln})
    return pd.DataFrame(rows)


# =========================
# RUN
# =========================
text_raw = open(TXT_PATH, "r", encoding="utf-8").read()
pages = preprocess_document(text_raw)

print("=== Textbook quick check ===")
print("Pages:", len(pages))
print("First page:", pages[0].page_num, "module_id:", pages[0].module_id)
print("\ntext_clean sample:\n", pages[0].text_clean[:250])
print("\ntext_en sample:\n", pages[0].text_en[:250])

# --- METRICS ---
by_page = compute_metrics_df_by_page(pages)
by_module = compute_metrics_df_by_module(by_page)
print("\n=== metrics_by_module ===")
display(by_module)

# --- CEFR ---
cefr_word_table = pd.DataFrame(columns=["word", "level", "frequency", "pos"])
if ENABLE_CEFR:
    word_levels = load_cefr_lexicon(CEFR_CSV_PATH)
    cefr_word_table, cefr_summary = compute_cefr_word_table(pages, word_levels, nlp)
    print("\n=== CEFR distribution (by tokens) ===")
    display(pd.DataFrame([cefr_summary["by_level_pct"]]))

# --- LEV ---
lev_words = pd.DataFrame(
    columns=["word", "frequency", "translation_ru", "translation_en_translit", "similarity"]
)
if ENABLE_LEV:
    lev_words, lev_summary = compute_lev_words_all(pages, nlp, translate_limit=LEV_TRANSLATE_TOP_N)
    print("\n=== Levenshtein summary ===")
    display(pd.DataFrame([lev_summary]))

# --- EX TYPES ---
exercises_df = pd.DataFrame(
    columns=["page_num", "module_id", "instruction", "lang", "chosen", "pred_label", "pred_conf"]
)

bundle_en_dir, bundle_ru_dir = resolve_model_bundle_dirs(
    str(MODEL_BUNDLE_EN_DIR), str(MODEL_BUNDLE_RU_DIR), REPO_ROOT
)
if bundle_en_dir and bundle_ru_dir:
    try:
        en_bundle = load_sklearn_bundle(bundle_en_dir, "EN")
        ru_bundle = load_sklearn_bundle(bundle_ru_dir, "RU")
        print(
            "\nExercise-type models loaded:",
            ("EN pipeline" if en_bundle[2] else "EN vect+clf"),
            "|",
            ("RU pipeline" if ru_bundle[2] else "RU vect+clf"),
        )
        ex_df = extract_candidate_instructions(pages)
        print("\n=== EX TYPES: candidates ===")
        print("Rows:", len(ex_df))
        display(ex_df.head(20))
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
        print("\n=== EX TYPES: predictions sample ===")
        display(exercises_df.head(30))
        if len(exercises_df) > 0:
            print("\n=== EX TYPES: label distribution ===")
            display(
                exercises_df["pred_label"]
                .value_counts()
                .reset_index()
                .rename(columns={"index": "label", "pred_label": "count"})
            )
    except Exception as e:
        print("\nEX TYPES failed:", repr(e))
else:
    print("\nEX TYPES skipped: model_bundle/model_bundle_ru not found.")

# --- OUTPUT: ONLY 3 CSV ---
OUTPUT_CEFR = OUTPUT_DIR / "cefr_word_table.csv"
OUTPUT_LEV = OUTPUT_DIR / "lev_words.csv"
OUTPUT_EX = OUTPUT_DIR / "exercises.csv"

cefr_word_table.to_csv(OUTPUT_CEFR, index=False)
lev_words.to_csv(OUTPUT_LEV, index=False)
exercises_df.to_csv(OUTPUT_EX, index=False)

print("\nSaved:")
print("-", OUTPUT_CEFR)
print("-", OUTPUT_LEV)
print("-", OUTPUT_EX)

print("\n=== OUTPUT PREVIEW: cefr_word_table (head 10) ===")
display(cefr_word_table.head(10))
print("\n=== OUTPUT PREVIEW: lev_words (head 10) ===")
display(lev_words.head(10))
print("\n=== OUTPUT PREVIEW: exercises (head 10) ===")
display(exercises_df.head(10))
