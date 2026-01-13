from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import re


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


def preprocess_document(text_raw: str, page_re: str = PAGE_RE_DEFAULT) -> List[PageBlock]:
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
