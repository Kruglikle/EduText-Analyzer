from pathlib import Path

from all_metrics import cefr


def test_load_cefr_lexicon_dedup(tmp_path: Path):
    csv_content = "\n".join(
        [
            "Word,CEFR Level",
            "apple,A2",
            "apple,B1",
            "banana,C1",
            "banana,A1",
            "cat,B2",
        ]
    )
    path = tmp_path / "cefr.csv"
    path.write_text(csv_content, encoding="utf-8")

    levels = cefr.load_cefr_lexicon(str(path))
    assert levels["apple"] == "A2"
    assert levels["banana"] == "A1"
    assert levels["cat"] == "B2"


def test_load_cefr_lexicon_cached(tmp_path: Path, monkeypatch):
    calls = {"count": 0}

    def fake_loader(path: str):
        calls["count"] += 1
        return {"word": "A1"}

    monkeypatch.setattr(cefr, "_load_cefr_lexicon_uncached", fake_loader)
    cefr._load_cefr_lexicon_cached.cache_clear()

    path = tmp_path / "cefr.csv"
    path.write_text("Word,CEFR Level\nword,A1\n", encoding="utf-8")

    first = cefr.load_cefr_lexicon(str(path))
    second = cefr.load_cefr_lexicon(str(path))

    assert calls["count"] == 1
    assert first == second
