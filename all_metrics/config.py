from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import os
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "len_cefr.csv").exists():
            return p
    return start


def _resolve_path(repo_root: Path, value: Optional[str], default: Path) -> Path:
    if value:
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        return p
    return default.resolve()


def _get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_str_env(name: str, default: str) -> str:
    return os.environ.get(name) or default


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    output_dir: Path
    runs_dir: Path
    cache_dir: Path
    translate_cache_path: Path
    cefr_csv_path: Path
    model_bundle_en_dir: Path
    model_bundle_ru_dir: Path
    translate_limit: int
    translate_sleep_sec: float
    translate_retries: int
    translate_timeout_sec: int
    log_level: str


def load_settings() -> Settings:
    repo_root = find_repo_root(Path.cwd())
    if load_dotenv:
        load_dotenv(repo_root / ".env")
    output_dir = _resolve_path(
        repo_root, os.environ.get("EDUTEXT_OUTPUT_DIR"), repo_root / "all_metrics"
    )
    runs_dir = _resolve_path(
        repo_root, os.environ.get("EDUTEXT_RUNS_DIR"), output_dir / "runs"
    )
    cache_dir = _resolve_path(
        repo_root, os.environ.get("EDUTEXT_CACHE_DIR"), output_dir / "cache"
    )
    translate_cache_path = _resolve_path(
        repo_root,
        os.environ.get("EDUTEXT_TRANSLATE_CACHE"),
        cache_dir / "translate_cache.csv",
    )
    cefr_csv_path = _resolve_path(
        repo_root, os.environ.get("EDUTEXT_CEFR_PATH"), repo_root / "len_cefr.csv"
    )
    model_bundle_en_dir = _resolve_path(
        repo_root, os.environ.get("EDUTEXT_MODEL_BUNDLE_EN"), repo_root / "model_bundle"
    )
    model_bundle_ru_dir = _resolve_path(
        repo_root, os.environ.get("EDUTEXT_MODEL_BUNDLE_RU"), repo_root / "model_bundle_ru"
    )

    translate_limit = _get_int_env("EDUTEXT_TRANSLATE_LIMIT", 300)
    translate_sleep_sec = _get_float_env("EDUTEXT_TRANSLATE_SLEEP_SEC", 0.2)
    translate_retries = _get_int_env("EDUTEXT_TRANSLATE_RETRIES", 2)
    translate_timeout_sec = _get_int_env("EDUTEXT_TRANSLATE_TIMEOUT_SEC", 10)
    log_level = _get_str_env("EDUTEXT_LOG_LEVEL", "INFO")

    return Settings(
        repo_root=repo_root,
        output_dir=output_dir,
        runs_dir=runs_dir,
        cache_dir=cache_dir,
        translate_cache_path=translate_cache_path,
        cefr_csv_path=cefr_csv_path,
        model_bundle_en_dir=model_bundle_en_dir,
        model_bundle_ru_dir=model_bundle_ru_dir,
        translate_limit=translate_limit,
        translate_sleep_sec=translate_sleep_sec,
        translate_retries=translate_retries,
        translate_timeout_sec=translate_timeout_sec,
        log_level=log_level,
    )


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
