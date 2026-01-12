#!/usr/bin/env python3
"""
CLI for classifying textbook exercises (communicative vs language) from OCR txt files.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import joblib


INSTRUCTION_WORDS = {
    "listen",
    "read",
    "write",
    "speak",
    "match",
    "complete",
    "choose",
    "fill",
    "ask",
    "say",
    "look",
    "circle",
    "tick",
    "underline",
    "copy",
    "colour",
    "color",
    "draw",
    "repeat",
    "answer",
    "послушай",
    "прочитай",
    "напиши",
    "скажи",
    "соедини",
    "вставь",
    "выбери",
    "отметь",
    "заполни",
    "составь",
    "раскрась",
    "нарисуй",
    "повтори",
    "ответь",
    "поговори",
}

NON_EXERCISE_PHRASES = {
    "in this module",
    "in this unit",
    "in this lesson",
    "you will learn",
    "learn, read and talk about",
    "now i know",
    "contents",
    "content",
    "module summary",
    "language portfolio",
    "grammar reference",
    "progress check",
    "word list",
}

NON_EXERCISE_MARKERS = {
    "isbn",
    "все права защищены",
    "выпускающий редактор",
    "корректоры",
    "подписано в печать",
    "формат",
    "тираж",
    "заказ",
    "express publishing",
    "colour illustrations",
    "illustrations",
    "tel:",
    "fax:",
}

PAGE_PATTERNS = [
    re.compile(r"[-=]{3,}\s*page\s+(\d+)\s*[-=]{3,}", re.IGNORECASE),
    re.compile(r"[-=]{3,}\s*страница\s+(\d+)\s*[-=]{3,}", re.IGNORECASE),
]

SECTION_PATTERN = re.compile(r"^\s*(module|unit|lesson)\s+(\d+)", re.IGNORECASE)
WORD_PATTERN = re.compile(r"[a-zа-яё]+", re.IGNORECASE)
NUMBERED_LINE_PATTERN = re.compile(r"(?m)^\s*\d+[\).]?\s+")
EXPLICIT_START_PATTERN = re.compile(r"^\s*(ex\.?|exercise|task|activity)\s*\d+", re.IGNORECASE)
EXPLICIT_LABEL_PATTERN = re.compile(r"(?m)^\s*(ex\.?|exercise|task|activity)\b", re.IGNORECASE)
PAGE_REF_PATTERN = re.compile(r"\bp\.\s*\d+\b", re.IGNORECASE)
SHORT_INSTRUCTION_MAX_WORDS = 12
SHORT_INSTRUCTION_LOOKAHEAD = 23


@dataclass
class Block:
    exercise_id: int
    page: Optional[int]
    section: Optional[str]
    lines: List[str]
    raw_text: str


class ModelWrapper:
    def __init__(self, pipeline=None, vectorizer=None, model=None):
        self.pipeline = pipeline
        self.vectorizer = vectorizer
        self.model = model
        if pipeline is not None:
            classes = getattr(pipeline, "classes_", None)
        elif model is not None:
            classes = getattr(model, "classes_", None)
        else:
            classes = None
        self.classes_ = classes if classes is not None else []

    def predict_proba(self, texts: List[str]):
        if self.pipeline is not None:
            return self.pipeline.predict_proba(texts)
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("Model and vectorizer must be provided when pipeline is absent.")
        features = self.vectorizer.transform(texts)
        return self.model.predict_proba(features)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u0400-\u04FF]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_page(line: str) -> Optional[int]:
    for pattern in PAGE_PATTERNS:
        match = pattern.search(line)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def detect_section(line: str) -> Optional[str]:
    match = SECTION_PATTERN.match(line)
    if match:
        title = match.group(0).strip()
        return title
    return None


def extract_tokens(text: str) -> List[str]:
    return WORD_PATTERN.findall(text.lower())


def instruction_word_in_prefix(text: str, max_words: int = 4) -> bool:
    tokens = extract_tokens(text)
    return any(token in INSTRUCTION_WORDS for token in tokens[:max_words])


def has_task_marker(text: str) -> bool:
    return bool(NUMBERED_LINE_PATTERN.search(text) or EXPLICIT_LABEL_PATTERN.search(text))


def is_short_instruction_line(line: str) -> bool:
    tokens = extract_tokens(line)
    if not tokens or len(tokens) > SHORT_INSTRUCTION_MAX_WORDS:
        return False
    return instruction_word_in_prefix(line)


def is_task_like(text: str) -> bool:
    if has_task_marker(text):
        return True
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return is_short_instruction_line(stripped)
    return False


def is_non_exercise_block(lines: List[str], text: str) -> bool:
    lower = text.lower()
    if any(marker in lower for marker in NON_EXERCISE_PHRASES):
        return True
    if any(marker in lower for marker in NON_EXERCISE_MARKERS):
        return True
    if "@" in text:
        return True

    page_refs = PAGE_REF_PATTERN.findall(text)
    if len(page_refs) >= 2:
        return True

    page_ref_lines = [ln for ln in lines if PAGE_REF_PATTERN.search(ln)]
    if len(page_ref_lines) >= 2:
        short_page_lines = [ln for ln in page_ref_lines if len(ln.strip()) <= 60]
        if len(short_page_lines) >= max(2, len(page_ref_lines) // 2):
            return True
    return False


def is_numbered_start_line(line: str) -> bool:
    return bool(NUMBERED_LINE_PATTERN.match(line) or EXPLICIT_START_PATTERN.match(line))


def is_start_line(line: str, page_has_numbered: bool, has_numbered_ahead: bool) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if any(marker in lower for marker in NON_EXERCISE_PHRASES):
        return False
    if is_numbered_start_line(stripped):
        return True
    if is_short_instruction_line(stripped) and (page_has_numbered or has_numbered_ahead):
        return True
    return False


def looks_like_exercise(lines: List[str]) -> bool:
    if not lines:
        return False
    raw_text = "\n".join(lines).strip()
    if not raw_text:
        return False
    if is_non_exercise_block(lines, raw_text):
        return False
    return is_task_like(raw_text)


def split_into_blocks(text: str) -> Tuple[List[Block], List[dict]]:
    normalized = normalize_text(text)
    lines = normalized.split("\n")
    current_lines: List[str] = []
    blocks: List[Block] = []
    filtered_out: List[dict] = []
    exercise_id = 1
    current_page: Optional[int] = None
    current_section: Optional[str] = None
    page_has_numbered = False

    def finalize_block():
        nonlocal exercise_id, current_lines
        if not current_lines:
            return
        raw_text = "\n".join(current_lines).strip()
        if not raw_text:
            current_lines = []
            return
        if looks_like_exercise(current_lines):
            blocks.append(
                Block(
                    exercise_id=exercise_id,
                    page=current_page,
                    section=current_section,
                    lines=current_lines.copy(),
                    raw_text=raw_text,
                )
            )
            exercise_id += 1
        else:
            filtered_out.append(
                {
                    "page": current_page,
                    "section": current_section,
                    "text": raw_text,
                }
            )
        current_lines = []

    def has_numbered_ahead(start_idx: int) -> bool:
        for offset in range(1, SHORT_INSTRUCTION_LOOKAHEAD + 1):
            idx = start_idx + offset
            if idx >= len(lines):
                return False
            if detect_page(lines[idx]) is not None:
                return False
            if detect_section(lines[idx]):
                return False
            if is_numbered_start_line(lines[idx]):
                return True
        return False

    for idx, line in enumerate(lines):
        page = detect_page(line)
        if page is not None:
            finalize_block()
            current_page = page
            page_has_numbered = False
            continue

        section = detect_section(line)
        if section:
            finalize_block()
            current_section = section
            continue

        has_ahead = False
        if not page_has_numbered and is_short_instruction_line(line):
            has_ahead = has_numbered_ahead(idx)
        if is_start_line(line, page_has_numbered, has_ahead):
            finalize_block()
            current_lines = [line]
            if is_numbered_start_line(line):
                page_has_numbered = True
            continue

        if current_lines:
            current_lines.append(line)
        elif line.strip():
            current_lines = [line]

    finalize_block()
    return blocks, filtered_out


def best_prompt(lines: List[str], model: ModelWrapper) -> Tuple[str, str, float]:
    non_empty = [ln.strip() for ln in lines if ln.strip()]
    if not non_empty:
        return "", "", 0.0

    candidates: List[str] = []
    first_candidate = "\n".join(non_empty[:8])
    candidates.append(first_candidate)

    window_size = 13
    if len(non_empty) <= window_size:
        candidates.append("\n".join(non_empty))
    else:
        for idx in range(0, len(non_empty) - window_size + 1):
            window = "\n".join(non_empty[idx : idx + window_size])
            candidates.append(window)

    seen = set()
    unique_candidates = []
    for cand in candidates:
        cand_stripped = cand.strip()
        if cand_stripped and cand_stripped not in seen:
            unique_candidates.append(cand_stripped)
            seen.add(cand_stripped)

    best_text = ""
    best_label = ""
    best_confidence = -1.0
    for cand in unique_candidates:
        probs = model.predict_proba([cand])[0]
        idx = probs.argmax()
        confidence = float(probs[idx])
        if confidence > best_confidence:
            best_confidence = confidence
            classes = getattr(model, "classes_", [])
            if classes is not None and len(classes):
                best_label = str(classes[idx])
            else:
                best_label = ""
            best_text = cand

    return best_text, best_label, best_confidence


def classify_blocks(
    blocks: List[Block], model: ModelWrapper, min_confidence: float
) -> Tuple[List[dict], int]:
    predictions = []
    review_count = 0
    for block in blocks:
        prompt, label, confidence = best_prompt(block.lines, model)
        review = confidence < min_confidence
        if review:
            review_count += 1
        predictions.append(
            {
                "exercise_id": block.exercise_id,
                "page": block.page,
                "section": block.section,
                "prompt": prompt,
                "label": label,
                "confidence": confidence,
                "review": review,
                "raw_text": block.raw_text,
            }
        )
    return predictions, review_count


def write_reports(
    predictions: List[dict],
    summary: List[dict],
    output_dir: Path,
    stem: str,
    dump_blocks: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{stem}_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "exercise_id",
                "page",
                "section",
                "prompt",
                "label",
                "confidence",
                "review",
            ],
        )
        writer.writeheader()
        for row in predictions:
            writer.writerow(
                {
                    "exercise_id": row["exercise_id"],
                    "page": row["page"],
                    "section": row["section"],
                    "prompt": row["prompt"],
                    "label": row["label"],
                    "confidence": f"{row['confidence']:.4f}",
                    "review": row["review"],
                }
            )

    summary_path = output_dir / f"{stem}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["type", "label", "page", "section", "count"]
        )
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    if dump_blocks:
        blocks_path = output_dir / f"{stem}_blocks.csv"
        with blocks_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["exercise_id", "page", "section", "raw_text"]
            )
            writer.writeheader()
            for row in predictions:
                writer.writerow(
                    {
                        "exercise_id": row["exercise_id"],
                        "page": row["page"],
                        "section": row["section"],
                        "raw_text": row["raw_text"],
                    }
                )


def build_summary(predictions: List[dict]) -> List[dict]:
    summary_rows: List[dict] = []
    label_counts = Counter(row["label"] for row in predictions)
    page_counts: defaultdict = defaultdict(Counter)
    section_counts: defaultdict = defaultdict(Counter)

    for row in predictions:
        page_key = row["page"] if row["page"] is not None else "unknown"
        page_counts[page_key][row["label"]] += 1
        if row.get("section"):
            section_counts[row["section"]][row["label"]] += 1

    for label, count in label_counts.items():
        summary_rows.append(
            {"type": "total_by_label", "label": label, "page": "", "section": "", "count": count}
        )

    for page, counter in sorted(page_counts.items(), key=lambda x: str(x[0])):
        for label, count in counter.items():
            summary_rows.append(
                {"type": "page_by_label", "label": label, "page": page, "section": "", "count": count}
            )

    for section, counter in section_counts.items():
        for label, count in counter.items():
            summary_rows.append(
                {
                    "type": "section_by_label",
                    "label": label,
                    "page": "",
                    "section": section,
                    "count": count,
                }
            )

    return summary_rows


def load_model(
    pipeline_path: Optional[Path],
    vectorizer_path: Optional[Path],
    model_path: Optional[Path],
) -> ModelWrapper:
    if pipeline_path and pipeline_path.exists():
        pipeline = joblib.load(pipeline_path)
        return ModelWrapper(pipeline=pipeline)

    if pipeline_path and not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

    if not vectorizer_path or not model_path:
        raise ValueError("Vectorizer and model paths are required when pipeline is not provided.")

    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return ModelWrapper(vectorizer=vectorizer, model=model)


def read_text_file(path: Path) -> str:
    encodings = ["utf-8", "utf-16", "cp1251", "latin-1"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def collect_text_files(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".txt":
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.txt"))
    raise FileNotFoundError(f"Input path not found or not a .txt file: {input_path}")


def print_sanity_check(
    predictions: List[dict],
    filtered: List[dict],
    blocks_total: int,
    review_count: int,
):
    print("\nSample predictions (first 30):")
    for row in predictions[:30]:
        snippet = row["prompt"].replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        page = row["page"] if row["page"] is not None else "-"
        print(f"page={page} | {snippet} | {row['label']} | {row['confidence']:.2f}")

    if filtered:
        print("\nFiltered (non-exercise) samples (first 10):")
        for sample in filtered[:10]:
            snippet = sample["text"].replace("\n", " ")
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            page = sample["page"] if sample["page"] is not None else "-"
            print(f"page={page} | {snippet}")

    print("\nSanity stats:")
    print(f"  blocks_total: {blocks_total}")
    print(f"  blocks_filtered: {len(filtered)}")
    print(f"  blocks_kept: {len(predictions)}")
    print(f"  review_count: {review_count}")


def process_file(
    path: Path,
    model: ModelWrapper,
    output_dir: Path,
    min_confidence: float,
    dump_blocks: bool,
):
    raw_text = read_text_file(path)
    blocks, filtered = split_into_blocks(raw_text)
    predictions, review_count = classify_blocks(blocks, model, min_confidence)
    summary = build_summary(predictions)
    write_reports(predictions, summary, output_dir, path.stem, dump_blocks=dump_blocks)
    blocks_total = len(blocks) + len(filtered)

    print(f"\nProcessed {path.name}:")
    print_sanity_check(predictions, filtered, blocks_total, review_count)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify textbook exercises.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to .txt file or directory containing .txt files.",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Directory to store prediction and summary CSV files (default: outputs).",
    )
    parser.add_argument(
        "--pipeline",
        default=None,
        help="Path to best_en_pipeline.pkl (sklearn Pipeline). If provided and exists, vectorizer/model args are ignored.",
    )
    parser.add_argument(
        "--vectorizer",
        default="tfidf_vectorizer.pkl",
        help="Path to tfidf_vectorizer.pkl (used when --pipeline is not provided).",
    )
    parser.add_argument(
        "--model",
        default="logreg_model (1).pkl",
        help="Path to logistic regression model pickle (used when --pipeline is not provided).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        dest="min_confidence",
        help="Threshold for review flagging (default: 0.60).",
    )
    parser.add_argument(
        "--dump-blocks",
        action="store_true",
        help="Save raw segmented blocks to CSV for manual inspection.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None):
    args = parse_args(argv)
    input_path = Path(args.input)
    output_dir = Path(args.output)
    pipeline_path = Path(args.pipeline) if args.pipeline else None
    vectorizer_path = Path(args.vectorizer) if args.vectorizer else None
    model_path = Path(args.model) if args.model else None

    try:
        model = load_model(pipeline_path, vectorizer_path, model_path)
    except Exception as exc:
        sys.stderr.write(f"Failed to load model: {exc}\n")
        sys.exit(1)

    try:
        text_files = collect_text_files(input_path)
    except Exception as exc:
        sys.stderr.write(f"Failed to collect input files: {exc}\n")
        sys.exit(1)

    if not text_files:
        sys.stderr.write("No .txt files found for processing.\n")
        sys.exit(1)

    for txt_file in text_files:
        process_file(
            txt_file,
            model=model,
            output_dir=output_dir,
            min_confidence=args.min_confidence,
            dump_blocks=args.dump_blocks,
        )


if __name__ == "__main__":
    main()
