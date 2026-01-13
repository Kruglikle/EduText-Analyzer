from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import logging
import os
import pickle
import re

import numpy as np
import pandas as pd

from .config import Settings


logger = logging.getLogger(__name__)

_MODEL_CACHE: Dict[str, Optional[Tuple[object, object, bool]]] = {
    "en": None,
    "ru": None,
}

_INSTRUCTION_RE_EN = re.compile(
    r"\b(Match|Complete|Choose|Read|Listen|Fill|Write|Discuss|Answer|Look|Put|Find|"
    r"Make|Work|Talk|Ask|Say|Tick|Circle|Underline|Label|Correct|Order|Number|Compare|"
    r"Describe|Explain|Practice|Guess|Use|Role-?play|Repeat|Translate|Spell|Draw|Mark|"
    r"Pair|Check|Think)\b",
    re.IGNORECASE,
)
_INSTRUCTION_RE_RU = re.compile(
    r"\b(Соотнеси|Заполни|Выбери|Прочитай|Прослушай|Напиши|Ответь|Скажи|Составь|"
    r"Найди|Поставь|Обсуди|Отметь|Подчеркни|Сравни|Опиши|Объясни|Слушай|Поговори|"
    r"Запиши|Укажи|Запомни|Сопоставь|Перепиши|Переведи|Произнеси|Повтори|Рассмотри|"
    r"Соедини|Допиши|Закончи)\b",
    re.IGNORECASE,
)
_INSTRUCTION_START_RE_EN = re.compile(
    r"^(?:(?:Now|Please|Let's|Lets)\s+)?"
    r"(Match|Complete|Choose|Read|Listen|Fill|Write|Discuss|Answer|Look|Put|Find|"
    r"Make|Work|Talk|Ask|Say|Tick|Circle|Underline|Label|Correct|Order|Number|Compare|"
    r"Describe|Explain|Practice|Guess|Use|Role-?play|Repeat|Translate|Spell|Draw|Mark|"
    r"Pair|Check|Think)\b",
    re.IGNORECASE,
)
_INSTRUCTION_START_RE_RU = re.compile(
    r"^(?:(?:Пожалуйста|Давайте|Теперь)\s+)?"
    r"(Соотнеси|Заполни|Выбери|Прочитай|Прослушай|Напиши|Ответь|Скажи|Составь|"
    r"Найди|Поставь|Обсуди|Отметь|Подчеркни|Сравни|Опиши|Объясни|Слушай|Поговори|"
    r"Запиши|Укажи|Запомни|Сопоставь|Перепиши|Переведи|Произнеси|Повтори|Рассмотри|"
    r"Соедини|Допиши|Закончи)\b",
    re.IGNORECASE,
)
_MAX_INSTRUCTION_LEN = 240
_MAX_INSTRUCTION_WORDS = 35
_PREFIX_RE = re.compile(r"^(?:\d+[\)\.]?\s*|\d+\s*[a-zA-Z]\)\s*|[a-zA-Z]\)\s*|[\-\*\u2022]\s+)")
_PUNCT_END_RE = re.compile(r"[.!?:…]$")


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


def resolve_model_bundle_dirs(settings: Settings) -> Tuple[Optional[str], Optional[str]]:
    en_path = str(settings.model_bundle_en_dir) if settings.model_bundle_en_dir.exists() else None
    ru_path = str(settings.model_bundle_ru_dir) if settings.model_bundle_ru_dir.exists() else None

    if en_path and ru_path:
        logger.info("Model bundles found in configured paths.")
        return en_path, ru_path

    logger.info("Searching for model bundles in: %s", settings.repo_root)
    if not en_path:
        en_path = _find_dir_by_name(str(settings.repo_root), "model_bundle")
    if not ru_path:
        ru_path = _find_dir_by_name(str(settings.repo_root), "model_bundle_ru")

    if en_path and ru_path:
        logger.info("Model bundles found via local search.")
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


def _load_pickle_any(path: str):
    try:
        import joblib

        obj = joblib.load(path)
        logger.info("Loaded %s via joblib", os.path.basename(path))
        return obj
    except Exception as joblib_exc:
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            logger.info("Loaded %s via pickle fallback", os.path.basename(path))
            return obj
        except Exception as pickle_exc:
            raise RuntimeError(
                f"Failed to load {path}. joblib={joblib_exc!r} pickle={pickle_exc!r}"
            )


def load_sklearn_bundle(bundle_dir: str, label: str) -> Tuple[object, object, bool]:
    if not os.path.isdir(bundle_dir):
        raise FileNotFoundError(f"{label}: bundle dir not found: {bundle_dir}")

    files = sorted(os.listdir(bundle_dir))
    logger.info("[%s] bundle dir: %s", label, bundle_dir)
    logger.info("[%s] files: %s", label, files)

    pkl_files = [f for f in files if f.lower().endswith(".pkl")]
    pipeline_candidates = [f for f in pkl_files if re.search(r"pipeline", f, re.IGNORECASE)]
    vectorizer_candidates = [f for f in pkl_files if re.search(r"(vectorizer|tfidf)", f, re.IGNORECASE)]
    model_candidates = [f for f in pkl_files if re.search(r"(model|clf|logreg)", f, re.IGNORECASE)]

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
        logger.info("[%s] loading mode: pipeline -> %s", label, os.path.basename(pipeline_path))
        try:
            pipe = _load_pickle_any(pipeline_path)
            return None, pipe, True
        except Exception as exc:
            logger.warning("[%s] pipeline load failed, will try vectorizer+clf: %s", label, exc)

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

    logger.info(
        "[%s] loading mode: vectorizer+clf -> %s + %s",
        label,
        os.path.basename(vect_path),
        os.path.basename(model_path),
    )
    try:
        vect = _load_pickle_any(vect_path)
        clf = _load_pickle_any(model_path)
        return vect, clf, False
    except Exception as exc:
        logger.error("[%s] vectorizer/model load failed: %s", label, exc)
        logger.error("[%s] dir contents: %s", label, files)
        raise


def load_models(settings: Settings) -> Tuple[Tuple[object, object, bool], Tuple[object, object, bool]]:
    if _MODEL_CACHE["en"] and _MODEL_CACHE["ru"]:
        return _MODEL_CACHE["en"], _MODEL_CACHE["ru"]

    en_dir, ru_dir = resolve_model_bundle_dirs(settings)
    if not en_dir or not ru_dir:
        raise FileNotFoundError("model_bundle and model_bundle_ru not found.")

    en_bundle = load_sklearn_bundle(en_dir, "EN")
    ru_bundle = load_sklearn_bundle(ru_dir, "RU")
    _MODEL_CACHE["en"] = en_bundle
    _MODEL_CACHE["ru"] = ru_bundle
    return en_bundle, ru_bundle


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


def _safe_conf(val) -> float:
    try:
        if val is None:
            return -1.0
        if isinstance(val, float) and np.isnan(val):
            return -1.0
        return float(val)
    except Exception:
        return -1.0


def choose_prediction(
    text: str,
    ru_pred,
    ru_conf,
    en_pred,
    en_conf,
) -> Tuple[object, float, str, str]:
    lang = detect_lang_simple(text)
    if lang == "ru":
        return ru_pred, float(ru_conf) if not np.isnan(ru_conf) else np.nan, "ru_lang", lang
    if lang == "en":
        return en_pred, float(en_conf) if not np.isnan(en_conf) else np.nan, "en_lang", lang

    rc = _safe_conf(ru_conf)
    ec = _safe_conf(en_conf)
    if rc >= ec:
        return ru_pred, rc if rc >= 0 else np.nan, "ru_conf", lang
    return en_pred, ec if ec >= 0 else np.nan, "en_conf", lang


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


def predict_ex_types_multilang(texts: List[str], ru_bundle, en_bundle) -> Dict[str, np.ndarray]:
    ru_vect, ru_clf, ru_is_pipe = ru_bundle
    en_vect, en_clf, en_is_pipe = en_bundle
    ru_pred, ru_conf = predict_bundle(texts, ru_vect, ru_clf, ru_is_pipe)
    en_pred, en_conf = predict_bundle(texts, en_vect, en_clf, en_is_pipe)
    final_pred, final_conf, chosen, lang_tag = [], [], [], []
    for i, t in enumerate(texts):
        pred, conf, ch, lang = choose_prediction(
            t, ru_pred[i], ru_conf[i], en_pred[i], en_conf[i]
        )
        final_pred.append(pred)
        final_conf.append(conf)
        chosen.append(ch)
        lang_tag.append(lang)
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
    rows = []
    for p in pages:
        lines = [ln.strip() for ln in (p.text_clean or "").splitlines()]
        for ln in lines:
            if not ln or len(ln) < 5:
                continue
            if len(ln) > _MAX_INSTRUCTION_LEN:
                continue
            if len(ln.split()) > _MAX_INSTRUCTION_WORDS:
                continue

            prefix_match = _PREFIX_RE.match(ln)
            stripped = _PREFIX_RE.sub("", ln).strip()
            start_match = _INSTRUCTION_START_RE_EN.match(stripped) or _INSTRUCTION_START_RE_RU.match(stripped)
            if not start_match:
                continue
            has_prefix = bool(prefix_match)
            if not has_prefix and not _PUNCT_END_RE.search(stripped) and "..." not in stripped:
                continue
            if has_prefix:
                prefix_text = (prefix_match.group(0) or "").strip()
                is_bullet = prefix_text.startswith(("-", "*", "\u2022"))
                is_letter = bool(re.match(r"^[A-Za-z]\)", prefix_text))
                if is_bullet and not is_letter and not _PUNCT_END_RE.search(stripped) and "..." not in stripped:
                    continue

            rows.append({"page_num": p.page_num, "module_id": p.module_id, "instruction": ln})
    return pd.DataFrame(rows)
